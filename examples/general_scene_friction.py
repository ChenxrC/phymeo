"""Declared physical inputs for the general scene friction acceptance cases.

UNEXECUTED tests-first. This module builds real scenes and explicit material
registrations. It does not implement an integrator, alter an accepted state, or
infer a persistent station token from the order of sampled contact rows.
"""

import gzip
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from itertools import product
from pathlib import Path
from types import MappingProxyType

import numpy as np

from phymeo.builders.urdf import load_urdf
from phymeo.constraints.collision_shapes import CollisionModel
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


def _paddle_asset(directory, body, sign, *, transformed=False, separated=False):
    path = Path(directory) / f"paddle-{body}.urdf"
    z = sign * (BOX_HALF[2] + RADIUS + (0.01 if separated else -DELTA))
    rotation, shift = placement(transformed)
    origin = vector_text(shift + rotation @ np.array([0.0, 0.0, z]))
    rpy = "1.5707963267948966 0 1.5707963267948966" if transformed else "0 0 0"
    path.write_text(
        f'''<robot name="paddle_{body}"><link name="base"/>
<link name="tool"><inertial><mass value="2"/>
<inertia ixx=".01" iyy=".015" izz=".02" ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="ball"><geometry><sphere radius="{RADIUS:.17g}"/></geometry></collision></link>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
<origin xyz="{origin}" rpy="{rpy}"/><axis xyz="1 0 0"/>
<limit lower="-.1" upper=".1" effort="8" velocity="8"/></joint></robot>''',
        encoding="utf-8",
    )
    return path


def make_two_paddles(
    directory,
    *,
    world=WORLD,
    friction=0.5,
    reverse=False,
    transformed=False,
    selected_tools=None,
    box_velocity_x=0.0,
    separated=False,
):
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
            _paddle_asset(directory, body, sign, transformed=transformed, separated=separated),
            robot_state=RobotState(np.zeros(1), np.zeros(1)),
            kp=[0.0],
            kd=[0.0],
            effort_limit=[8.0],
            world=world,
        )
    rotation, shift = placement(transformed)
    builder.add_box(
        BOX,
        BOX_HALF,
        mass=1.0,
        position=shift,
        orientation_xyzw=(0.5,) * 4 if transformed else (0, 0, 0, 1),
        velocity=rotation @ np.array([box_velocity_x, 0.0, 0.0]),
        world=world,
    )
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
        if selected_tools is not None:
            selected = frozenset(selected_tools)
            if not selected <= {LOWER_TOOL, UPPER_TOOL}:
                raise ValueError("selected_tools must name declared paddle bodies")
            allowed = {101 if body == LOWER_TOOL else 202 for body in selected}
            registrations = [r for r in registrations if r.key.token in allowed]
        registry = MaterialStationRegistry(registrations)
        actions = MappingProxyType(
            {body: RobotCommand([0.0], feedforward=[0.0]) for body, _ in declarations}
        )
        parameters = MappingProxyType(
            dict(
                case="F01",
                dt_s=H,
                stiffness_n_per_m=KT,
                friction=friction,
                penetration_m=DELTA,
                rotation=tuple(map(tuple, rotation)),
                shift=tuple(shift),
                separated=separated,
                box_velocity_x=box_velocity_x,
            )
        )
        return FrictionCase(scene, registry, actions, world, parameters)
    except BaseException:
        scene.close()
        raise


CYCLE = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
SHIFT = np.array([0.3, -0.2, 0.4])


def placement(transformed=False):
    return (CYCLE.copy(), SHIFT.copy()) if transformed else (np.eye(3), np.zeros(3))


def vector_text(values):
    return " ".join(format(float(v), ".17g") for v in values)


def _inertia(mass, values=(0.01, 0.015, 0.02)):
    x, y, z = values
    return (
        f'<inertial><mass value="{mass}"/>'
        f'<inertia ixx="{x}" iyy="{y}" izz="{z}" ixy="0" ixz="0" iyz="0"/>'
        "</inertial>"
    )


def _write_asset(directory, name, content):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.urdf"
    path.write_text(f'<robot name="{name}">{content}</robot>', encoding="utf-8")
    return path


def _single_asset(
    directory,
    name,
    *,
    mass,
    origin,
    geometry,
    axis=(1, 0, 0),
    kind="prismatic",
    inertia=(0.01, 0.015, 0.02),
    limit=0.1,
    effort=8,
    speed=8,
):
    content = (
        '<link name="base"/><link name="tool">'
        + _inertia(mass, inertia)
        + f'<collision name="contact"><geometry>{geometry}</geometry></collision></link>'
        f'<joint name="drive" type="{kind}"><parent link="base"/><child link="tool"/>'
        f'<origin xyz="{vector_text(origin)}"/><axis xyz="{vector_text(axis)}"/>'
        f'<limit lower="{-limit}" upper="{limit}" effort="{effort}" velocity="{speed}"/>'
        "</joint>"
    )
    return _write_asset(directory, name, content)


def _registration(world, token, anchor_key, other_key, anchor, anchor_feature, other_feature):
    first, second = sorted((anchor_key, other_key))
    anchor_first = first == anchor_key
    return MaterialStationRegistration(
        StationKey(world, token),
        first,
        second,
        "first" if anchor_first else "second",
        tuple(anchor),
        (1.0, 0.0, 0.0),
        1000 + token if anchor_first else 2000 + token,
        2000 + token if anchor_first else 1000 + token,
        first_feature=anchor_feature if anchor_first else other_feature,
        second_feature=other_feature if anchor_first else anchor_feature,
    )


def _finish_case(scene, registrations, actions, world, **parameters):
    try:
        return FrictionCase(
            scene,
            MaterialStationRegistry(registrations),
            MappingProxyType(dict(actions)),
            world,
            MappingProxyType(dict(parameters, stiffness_n_per_m=KT)),
        )
    except BaseException:
        scene.close()
        raise


def make_rod(directory, *, friction=0.5, center_y=0.0, drive=0.5, world=WORLD):
    """F02: real capsule axis against the finite face of a non-box convex solid."""
    path = _single_asset(
        directory,
        "rod",
        mass=2,
        origin=(0, center_y, 1 / 32 - DELTA),
        geometry='<sphere radius=".03125"/>',
    )
    asset = load_urdf(path, gravity=(0, 0, 0))
    original = asset.collisions[0]
    shape = replace(original.shape, kind="capsule", vertex_ids=(0, 1), radius=1 / 32)
    collider = replace(original, shape=shape, local_vertices=((0, -1 / 8, 0), (0, 1 / 8, 0)))
    asset = replace(asset, collisions=(collider,), collision_model=CollisionModel((shape,)))

    def loader(path, *, gravity):
        return asset

    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    builder.add_urdf(
        LOWER_TOOL,
        path,
        robot_state=RobotState(np.zeros(1), np.zeros(1)),
        kp=[0],
        kd=[0],
        loader=loader,
        world=world,
    )
    vertices = tuple(
        (sx * (0.3 if z < 0 else 0.25), sy * (0.2 if z < 0 else 0.15), z)
        for sx, sy, z in product((-1, 1), (-1, 1), (-0.1, 0.0))
    )
    builder.add_convex(BOX, vertices, mass=None, world=world)
    builder.pair(LOWER_TOOL, BOX, RobotContactMaterial(2048, 0, friction, 0.01), world=world)
    scene = builder.build(backend="host")
    rod = scene.shape_key(LOWER_TOOL, "tool", "contact")
    (table,) = scene.body_keys[world, BOX]
    registrations = [
        _registration(world, 210 + i, rod, table, (0, y, -1 / 32), (0, 1), (1, 3, 5, 7))
        for i, y in enumerate((-1 / 8, 1 / 8))
    ]
    return _finish_case(
        scene,
        registrations,
        {LOWER_TOOL: RobotCommand([0], feedforward=[drive])},
        world,
        case="F02",
        dt_s=H,
        friction=friction,
        center_y=center_y,
        drive=drive,
    )


def make_fixed_mimic_pair(
    directory, *, reverse=False, world=WORLD, partner_cap=8.0, gravity=(0, 0, 0)
):
    """F03: two affine-related spherical tools share a real third body's response."""
    from phymeo.builders.mimic_urdf import load_mimic_urdf

    parts = ['<link name="base"/>']
    for name, mass, y, mimic in (
        ("driver", 1, -1 / 8, ""),
        ("follower", 2, 1 / 8, '<mimic joint="driver" multiplier="2" offset="0"/>'),
    ):
        parts.append(
            f'<link name="{name}_tool">'
            + _inertia(mass)
            + '<collision name="ball"><geometry><sphere radius=".0625"/></geometry></collision></link>'
            f'<joint name="{name}" type="prismatic"><parent link="base"/><child link="{name}_tool"/>'
            f'<origin xyz="0 {y} {RADIUS - DELTA}"/><axis xyz="1 0 0"/>'
            '<limit lower="-.1" upper=".1" effort="8" velocity="8"/>' + mimic + "</joint>"
        )
    mimic_path = _write_asset(directory, "affine_tools", "".join(parts))
    plate_path = _single_asset(
        directory, "plate", mass=3, origin=(0, 0, -0.05), geometry='<box size=".6 .6 .1"/>'
    )
    builder = ContactSceneBuilder(gravity=gravity)

    def first():
        builder.add_urdf(
            LOWER_TOOL,
            mimic_path,
            robot_state=RobotState(np.zeros(1), np.zeros(1)),
            kp=[0],
            kd=[0],
            loader=load_mimic_urdf,
            world=world,
        )

    def second():
        builder.add_urdf(
            UPPER_TOOL,
            plate_path,
            robot_state=RobotState(np.zeros(1), np.zeros(1)),
            kp=[0],
            kd=[0],
            effort_limit=[partner_cap],
            world=world,
        )

    for add in (second, first) if reverse else (first, second):
        add()
    builder.pair(LOWER_TOOL, UPPER_TOOL, RobotContactMaterial(2048, 0, 0.5, 0.01), world=world)
    scene = builder.build(backend="host")
    plate = scene.shape_key(UPPER_TOOL, "tool", "contact")
    registrations = [
        _registration(
            world,
            310 + i,
            scene.shape_key(LOWER_TOOL, f"{name}_tool", "ball"),
            plate,
            (0, 0, -RADIUS),
            (0,),
            (1, 3, 5, 7),
        )
        for i, name in enumerate(("driver", "follower"))
    ]
    actions = {
        LOWER_TOOL: RobotCommand([0], feedforward=[0]),
        UPPER_TOOL: RobotCommand([0], feedforward=[1]),
    }
    return _finish_case(
        scene,
        registrations,
        actions,
        world,
        case="F03",
        dt_s=H,
        partner_cap=partner_cap,
        gravity=tuple(gravity),
    )


def make_floating_clamp(directory, *, fixture_dir, mimic=True, reverse=False, world=WORLD):
    """F04: unchanged positive-mass coaxial tree; only the declared affine constraint differs."""
    from xml.etree import ElementTree as ET

    from phymeo.builders.floating_mimic_urdf import load_floating_mimic_urdf
    from phymeo.builders.floating_urdf import load_floating_urdf
    from phymeo.dynamics.floating_robot import FloatingRobotState
    from phymeo.runtime._floating_robot_participant import FloatingRobotCommand

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    original = Path(fixture_dir) / "floating_mimic_jaws.urdf"
    if mimic:
        floating_path = original
    else:
        xml = ET.parse(original)
        for joint in xml.getroot().findall("joint"):
            for relation in joint.findall("mimic"):
                joint.remove(relation)
        floating_path = directory / "independent_jaws.urdf"
        xml.write(floating_path, encoding="utf-8", xml_declaration=True)
    box_collisions = "".join(
        f'<collision name="{name}"><origin xyz="0 0 {sign * (1 / 16 + 0.02 - DELTA):.17g}"/>'
        '<geometry><box size=".2 .2 .04"/></geometry></collision>'
        for name, sign in (("lower", -1), ("upper", 1))
    )
    plate_path = _write_asset(
        directory,
        "floating_partner",
        '<link name="base"/><link name="tool">' + _inertia(3) + box_collisions + "</link>"
        '<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>'
        '<origin xyz=".254 0 0"/><axis xyz="1 0 0"/>'
        '<limit lower="-.1" upper=".1" effort="8" velocity="8"/></joint>',
    )
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    joints = [0.0] if mimic else [0.0, 0.004]
    k = len(joints)

    def first():
        builder.add_floating_urdf(
            LOWER_TOOL,
            floating_path,
            robot_state=FloatingRobotState(
                np.array([0, 0, 0, 0, 0, 0, 1, *joints], float), np.zeros(6 + k)
            ),
            kp=np.zeros(k),
            kd=np.zeros(k),
            loader=load_floating_mimic_urdf if mimic else load_floating_urdf,
            world=world,
        )

    def second():
        builder.add_urdf(
            UPPER_TOOL,
            plate_path,
            robot_state=RobotState(np.zeros(1), np.zeros(1)),
            kp=[0],
            kd=[0],
            world=world,
        )

    for add in (second, first) if reverse else (first, second):
        add()
    builder.pair(LOWER_TOOL, UPPER_TOOL, RobotContactMaterial(2048, 0, 0.5, 0.01), world=world)
    scene = builder.build(backend="host")
    right = scene.shape_key(LOWER_TOOL, "right_jaw", "right_pad")
    registrations, anchors = [], []
    for side, (name, sign) in enumerate((("lower", -1), ("upper", 1))):
        plate = scene.shape_key(UPPER_TOOL, "tool", name)
        for corner, (x, y) in enumerate(product((-1 / 32, 1 / 32), (-1 / 16, 1 / 16))):
            token = 410 + side * 4 + corner
            point = (x, y, sign / 16)
            anchors.append(point)
            registrations.append(
                _registration(
                    world,
                    token,
                    right,
                    plate,
                    point,
                    (0, 2, 4, 6) if sign < 0 else (1, 3, 5, 7),
                    (1, 3, 5, 7) if sign < 0 else (0, 2, 4, 6),
                )
            )
    actions = {
        LOWER_TOOL: FloatingRobotCommand(
            RobotCommand(joints, feedforward=[0.1] if mimic else [0, 0.1])
        ),
        UPPER_TOOL: RobotCommand([0], feedforward=[0]),
    }
    return _finish_case(
        scene,
        registrations,
        actions,
        world,
        case="F04",
        dt_s=H,
        mimic=mimic,
        right_material_anchors=tuple(anchors),
    )


def _rectangle_sites(x, y):
    corners = np.array([[-x, -y], [x, -y], [x, y], [-x, y]])
    return tuple(
        tuple(wl * left + wr * right)
        for edge in range(4)
        for left, right in ((corners[edge], corners[(edge + 1) % 4]),)
        for wl, wr in ((1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3))
    )


def make_rotating_tool(directory, *, common=False, tilted=False, world=WORLD):
    """F05: finite rotating faces with a deliberately separate large-motion rejection."""
    if common and tilted:
        raise ValueError("the two frozen F05 holdouts are distinct")
    axis = (0, 1, 0) if tilted else (0, 0, 1)
    speed = 5.0 if tilted else 0.2 if common else 0.0
    h = 1 / 128 if tilted else 1 / 4096
    path = _single_asset(
        directory,
        "rotor",
        mass=1,
        origin=(0, 0, 0.03125 - DELTA),
        geometry='<box size=".25 .125 .0625"/>',
        axis=axis,
        kind="revolute",
        inertia=(0.025, 0.025, 0.02),
        limit=0.25,
        effort=0.05,
        speed=8 if tilted else 1,
    )
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    builder.add_urdf(
        LOWER_TOOL,
        path,
        robot_state=RobotState(np.zeros(1), np.array([speed])),
        kp=[0],
        kd=[0],
        world=world,
    )
    if common:
        other_path = _single_asset(
            directory,
            "co_rotor",
            mass=1,
            origin=(0, 0, -0.03125),
            geometry='<box size=".25 .125 .0625"/>',
            axis=(0, 0, 1),
            kind="revolute",
            inertia=(0.025, 0.025, 0.02),
            limit=0.25,
            effort=0.05,
            speed=1,
        )
        builder.add_urdf(
            UPPER_TOOL,
            other_path,
            robot_state=RobotState(np.zeros(1), np.array([0.2])),
            kp=[0],
            kd=[0],
            world=world,
        )
        other_body = UPPER_TOOL
    else:
        builder.add_box(BOX, (0.5, 0.5, 0.05), mass=None, position=(0, 0, -0.05), world=world)
        other_body = BOX
    builder.pair(LOWER_TOOL, other_body, RobotContactMaterial(2048, 0, 0.5, 0.01), world=world)
    scene = builder.build(backend="host")
    rotor = scene.shape_key(LOWER_TOOL, "tool", "contact")
    other = (
        scene.shape_key(UPPER_TOOL, "tool", "contact") if common else scene.body_keys[world, BOX][0]
    )
    xy = (
        tuple(product((-0.125, 0.125), (-0.0625, 0.0625)))
        if common
        else _rectangle_sites(0.125, 0.0625)
    )
    anchors = tuple((x, y, -0.03125) for x, y in xy)
    registrations = [
        _registration(world, 510 + i, rotor, other, anchor, (0, 2, 4, 6), (1, 3, 5, 7))
        for i, anchor in enumerate(anchors)
    ]
    actions = {LOWER_TOOL: RobotCommand([0], feedforward=[0 if common else 0.01])}
    if common:
        actions[UPPER_TOOL] = RobotCommand([0], feedforward=[0])
    return _finish_case(
        scene,
        registrations,
        actions,
        world,
        case="F05",
        dt_s=h,
        common=common,
        tilted=tilted,
        rotor_anchors=anchors,
        axis=axis,
    )


def make_withdrawal(directory, *, world=WORLD):
    """F06: prescribed motor forces, never prescribed accepted poses or a reset."""
    path = _write_asset(
        directory,
        "withdrawal",
        '<link name="base"/><link name="carriage">' + _inertia(1) + "</link>"
        '<link name="tool">'
        + _inertia(1)
        + '<collision name="face"><geometry><box size=".5 .5 .1"/></geometry></collision></link>'
        '<joint name="slide_x" type="prismatic"><parent link="base"/><child link="carriage"/>'
        '<axis xyz="1 0 0"/><limit lower="-.1" upper=".1" effort="8" velocity="8"/></joint>'
        '<joint name="slide_z" type="prismatic"><parent link="carriage"/><child link="tool"/>'
        '<origin xyz="0 0 -.05"/><axis xyz="0 0 1"/>'
        '<limit lower="-.1" upper=".1" effort="600" velocity="8"/></joint>',
    )
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    builder.add_urdf(
        LOWER_TOOL,
        path,
        robot_state=RobotState(np.zeros(2), np.zeros(2)),
        kp=[0, 0],
        kd=[0, 0],
        world=world,
    )
    builder.add_sphere(BOX, 1 / 8, mass=None, position=(0, 0, 1 / 8 - DELTA), world=world)
    builder.pair(LOWER_TOOL, BOX, RobotContactMaterial(2048, 0, 0.5, 0.01), world=world)
    scene = builder.build(backend="host")
    face, ball = scene.shape_key(LOWER_TOOL, "tool", "face"), scene.body_keys[world, BOX][0]
    registration = _registration(world, 610, ball, face, (0, 0, -1 / 8), (0,), (1, 3, 5, 7))
    return _finish_case(
        scene,
        [registration],
        withdrawal_actions(0),
        world,
        case="F06",
        dt_s=1 / 64,
        x_drive=(0.125, 0, 0),
        z_drive=(-254, 512, -254),
    )


def withdrawal_actions(index):
    x, z = (0.125, 0.0, 0.0)[index], (-254.0, 512.0, -254.0)[index]
    return {LOWER_TOOL: RobotCommand([0, 0], feedforward=[x, z])}


def public_payload(value):
    """Lossless JSON values for all public fields; private ownership seals stay opaque."""
    if isinstance(value, np.ndarray):
        return {"array": value.tolist(), "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        public_fields = {
            field.name: public_payload(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
        if "_generations" in value.__dataclass_fields__:
            public_fields["generations"] = public_payload(value.generations)
        return {
            "type": type(value).__module__ + "." + type(value).__qualname__,
            "fields": public_fields,
        }
    if isinstance(value, Mapping):
        return {
            "mapping": [
                [public_payload(key), public_payload(item)]
                for key, item in sorted(value.items(), key=lambda entry: repr(entry[0]))
            ]
        }
    if isinstance(value, (tuple, list)):
        return [public_payload(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {"set": [public_payload(item) for item in sorted(value, key=repr)]}
    if isinstance(value, Enum):
        return {"enum": type(value).__qualname__, "value": public_payload(value.value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unhandled public output type: {type(value).__qualname__}")


def capture_frame(case, entry, reading):
    """Read current owned observations only; this function does not advance a scene."""
    return dict(
        entry=entry,
        reading=reading,
        geometry=case.scene.geometry(),
        history=case.scene.tangent_history,
        receipt=case.scene.tangent_receipt,
        events=case.scene.contact_events,
    )


def write_trace(path, *, case, frames, failure=None, actual_dt_s=None):
    """Save every public field once without overwriting a prior raw artifact."""
    path = Path(path)
    parameters = dict(case.parameters)
    parameters["declared_dt_s"] = parameters["dt_s"]
    if actual_dt_s is not None:
        parameters["dt_s"] = float(actual_dt_s)
    payload = public_payload(
        dict(
            parameters=parameters,
            world=case.world,
            registrations=case.registry.registrations,
            declared_actions=case.actions,
            frames=tuple(frames),
            failure=failure,
        )
    )
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(
                json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
            )
    return path

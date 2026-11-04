"""A reusable named pad binds to material in each imported asset, even after declaration reorder."""

from dataclasses import replace

import numpy as np
import pytest

from phymeo.builders.urdf import load_urdf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.rigid import RigidBodyState
from phymeo.core.robot import RobotState
from phymeo.runtime.robot_payload_scene import PadFace, RobotPayloadScene


def asset(tmp_path, guard_first=False):
    jaw = '<collision name="jaw"><geometry><box size="0.006 0.024 0.02"/></geometry></collision>'
    guard = '<collision name="guard"><origin xyz="0 0.10 0"/><geometry><box size="0.006 0.024 0.02"/></geometry></collision>'
    sites = (guard + jaw) if guard_first else (jaw + guard)
    path = tmp_path / ("reordered.urdf" if guard_first else "original.urdf")
    path.write_text(
        '<robot name="rail"><link name="base"/><link name="tool"><inertial><mass value="1"/>'
        '<inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>'
        + sites
        + "</link>"
        '<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/><origin xyz="0 0 0.25"/>'
        '<axis xyz="1 0 0"/><limit lower="-0.1" upper="0.1" effort="8" velocity="1"/></joint></robot>'
    )
    return load_urdf(path, gravity=(0, 0, 0))


def scene(arm, pads):
    free = RigidBodyState(
        np.array([[0.022, 0, 0.25]]),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.zeros((1, 3)),
        np.zeros((1, 3)),
    )
    return RobotPayloadScene(
        arm,
        robot_state=RobotState(np.zeros(1), np.zeros(1)),
        payload_state=free,
        payload_geometry={"kind": "sphere", "radius_m": 0.020},
        mass_kg=0.125,
        table_center_m=(0, 0, -0.1),
        table_half_extents_m=(0.2, 0.2, 0.01),
        dt_s=0.001,
        kp=[0],
        kd=[0],
        pads=pads,
        effort_limit=[8.0],
        pad_material=RobotContactMaterial(2000, 0, 0),
        structure_material=RobotContactMaterial(2000, 0, 0),
    )


def test_named_face_is_resolved_against_each_urdf_and_applies_force_to_the_named_material(tmp_path):
    a, b = asset(tmp_path), asset(tmp_path, True)
    spec = PadFace.named("tool", "jaw", 0, 1, [(0, 0)])
    current = []
    for source in (a, b):
        key = next(c.shape.key for c in source.collisions if c.name == "jaw")
        reading = scene(source, [spec]).step([0])
        assert reading.pad_shape_keys == (key,) and reading.structural_contacts == ()
        np.testing.assert_allclose(reading.pad_points_m, [[0.003, 0, 0.25]], atol=1e-14)
        np.testing.assert_allclose(reading.pad_normal_loads_N, [2.0], atol=1e-13)
        np.testing.assert_allclose(reading.contact_effort, [-2.0], atol=1e-13)
        np.testing.assert_allclose(reading.payload_velocity_m_s, [0.016, 0, 0], atol=2e-14)
        current.append(key)
    assert current[0] != current[1]
    # Old explicit numeric selectors remain local. Reusing the first key on b
    # selects its guard; the physically touching jaw then reports a structural contact.
    stale = scene(b, [PadFace(current[0], 0, 1, [(0, 0)])]).step([0])
    np.testing.assert_array_equal(stale.pad_normal_loads_N, [0.0])
    assert [c["shape"] for c in stale.structural_contacts] == ["jaw"]


def test_named_face_rejects_missing_ambiguous_and_duplicate_material_channels(tmp_path):
    a = asset(tmp_path)
    jaw, guard = a.collisions
    renamed = replace(a, collisions=(jaw, replace(guard, name=jaw.name)))
    for target, pair, expected in (
        (a, ("wrong", "jaw"), "not found"),
        (a, ("tool", "missing"), "not found"),
        (renamed, ("tool", "jaw"), "ambiguous"),
    ):
        with pytest.raises(ValueError, match=expected):
            scene(target, [PadFace.named(*pair, 0, 1, [(0, 0)])])
    named = PadFace.named("tool", "jaw", 0, 1, [(0, 0)])
    with pytest.raises(ValueError, match="duplicate"):
        scene(a, [named, PadFace(jaw.shape.key, 0, 1, [(0, 0)])])


@pytest.mark.parametrize("format,generated", [("urdf", "collision_0"), ("mjcf", "geom_0")])
def test_an_authored_name_can_equal_an_ordinal_but_anonymous_material_cannot_bind_by_name(
    tmp_path, format, generated
):
    from xml.etree import ElementTree as ET

    from phymeo.builders.mjcf import load_mjcf

    for explicit in (False, True):
        if format == "urdf":
            asset(tmp_path)
            doc = ET.parse(tmp_path / "original.urdf")
            col = doc.find("./link[@name='tool']/collision")
            if explicit:
                col.set("name", generated)
            else:
                del col.attrib["name"]
            path = tmp_path / "ordinal.urdf"
            doc.write(path)
            source = load_urdf(path, gravity=(0, 0, 0))
        else:
            attr = f'name="{generated}"' if explicit else ""
            path = tmp_path / "ordinal.xml"
            path.write_text(
                '<mujoco model="rail"><compiler angle="radian" inertiafromgeom="false"/>'
                '<worldbody><body name="base"><body name="tool" pos="0 0 0.25">'
                '<joint name="slide" type="slide" axis="1 0 0" range="-0.1 0.1"/>'
                '<inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>'
                f'<geom {attr} type="box" size="0.003 0.012 0.01"/>'
                "</body></body></worldbody></mujoco>"
            )
            source = load_mjcf(path, gravity=(0, 0, 0))
        hit = next(c for c in source.collisions if c.name == generated)
        assert hit.name_is_authored is explicit
        spec = PadFace.named("tool", generated, 0, 1, [(0, 0)])
        if explicit:
            reading = scene(source, [spec]).step([0])
        else:
            with pytest.raises(ValueError, match="not found"):
                scene(source, [spec])
            reading = scene(source, [PadFace(hit.shape.key, 0, 1, [(0, 0)])]).step([0])
        np.testing.assert_allclose(reading.pad_normal_loads_N, [2.0], atol=2e-13)

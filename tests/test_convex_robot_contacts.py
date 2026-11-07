"""Convex families drive actual fixed and floating owners in one scene transaction."""
from dataclasses import replace
from itertools import combinations_with_replacement, product

import numpy as np
import pytest

from phymeo.builders.floating_urdf import load_floating_urdf
from phymeo.builders.urdf import load_urdf
from phymeo.constraints.collision_shapes import CollisionModel
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.dynamics.floating_robot import FloatingRobotState
from phymeo.runtime._floating_robot_participant import FloatingRobotCommand
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand

H = .001
KINDS = ("sphere", "capsule", "box", "hull")
PAIRS = tuple(combinations_with_replacement(KINDS, 2))
LAW = RobotContactMaterial(100, 0, 0)


def _vertices(kind):
    if kind == "sphere":
        return np.array([[0., 0., 0.]]), .05
    if kind == "capsule":
        return np.array([[-.5, 0, 0], [.5, 0, 0]]), .05
    if kind == "box":
        return np.array(list(product((-.6,.6),(-.1,.1),(-.1,.1)))), 0.
    ring = ((-.6,0),(-.4,-.1),(.4,-.1),(.6,0),(.4,.1),(-.4,.1))
    return np.array([(x,y,z) for x,y in ring for z in (-.1,.1)]), 0.


def _asset(path, kind, mass, floating, z):
    inertia = f'<inertial><mass value="{mass}"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>'
    collision = '<collision name="shape"><geometry><sphere radius=".05"/></geometry></collision>'
    if floating:
        source = f'<robot name="convex"><link name="base">{inertia}{collision}</link></robot>'
    else:
        source = f'<robot name="convex"><link name="base"/><link name="tool">{inertia}{collision}</link><joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/><origin xyz="0 0 {z}"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="128" velocity="10"/></joint></robot>'
    path.write_text(source, encoding="utf-8")
    asset = (load_floating_urdf if floating else load_urdf)(path, gravity=(0,0,0))
    original = asset.collisions[0]
    vertices, radius = _vertices(kind)
    shape = replace(original.shape, kind="hull" if kind == "box" else kind,
                    vertex_ids=tuple(range(17,17+len(vertices))), radius=radius)
    collider = replace(original, shape=shape, local_vertices=tuple(map(tuple, vertices)))
    return replace(asset, collisions=(collider,), collision_model=CollisionModel((shape,)))


def _scene(tmp_path, kinds, owners, gap=-.01, velocity=(0.,0.), reverse=False):
    builder = ContactSceneBuilder(gravity=(0,0,0))
    height = sum(.05 if k in ("sphere","capsule") else .1 for k in kinds) + gap
    actions = {}
    order = (1,0) if reverse else (0,1)
    for i in order:
        body, mass, z, floating = (10,20)[i], (2.,3.)[i], (0.,height)[i], owners[i]
        path = tmp_path/f"body{body}.urdf"
        asset = _asset(path,kinds[i],mass,floating,z)
        def loader(path, *, gravity, asset=asset):
            return asset
        if floating:
            builder.add_floating_urdf(body,path,
                robot_state=FloatingRobotState(np.array([0,0,z,0,0,0,1.]),np.array([0,0,velocity[i],0,0,0.])),
                kp=[],kd=[],loader=loader)
            actions[body] = FloatingRobotCommand(None)
        else:
            builder.add_urdf(body,path,robot_state=RobotState(np.zeros(1),np.array([velocity[i]])),
                             kp=[0],kd=[0],loader=loader)
            actions[body] = RobotCommand([0])
    builder.pair(10,20,LAW)
    return builder, actions


@pytest.mark.parametrize("kinds", PAIRS)
@pytest.mark.parametrize("owners", ((False,False),(False,True),(True,True)))
def test_family_patch_accelerates_both_actual_owners(tmp_path,kinds,owners):
    builder, actions = _scene(tmp_path,kinds,owners)
    with builder.build() as scene:
        out = scene.step(H,actions=actions)
        assert out.loads.contacts
        assert sum(row.force_n for row in out.loads.contacts) == pytest.approx(1.,abs=1e-10)
        for i, body in enumerate((10,20)):
            obs=out.robots[0,body]
            force=(-1.,1.)[i]
            np.testing.assert_allclose(obs.link_wrenches_world[0,:3],(0,0,force),atol=1e-10)
            assert obs.v[2 if owners[i] else 0] == pytest.approx(H*force/(2.,3.)[i],abs=1e-12)
            np.testing.assert_allclose(obs.fixed_root_contact_wrench,0,atol=1e-12)
        force=sum(np.array(row["force_on_first_N"])+row["force_on_second_N"] for row in out.contacts)
        np.testing.assert_allclose(force,0,atol=1e-12)


@pytest.mark.parametrize("kinds", (("capsule","capsule"),("capsule","box"),("box","hull")))
def test_separation_touch_velocity_and_registration_order(tmp_path,kinds):
    for gap in (.02,0.):
        folder=tmp_path/str(gap); folder.mkdir()
        builder,actions=_scene(folder,kinds,(False,True),gap=gap,velocity=(.1,-.2))
        with builder.build() as scene:
            out=scene.step(H,actions=actions)
            assert not out.loads.contacts
            assert out.robots[0,10].v[0] == pytest.approx(.1)
            assert out.robots[0,20].v[2] == pytest.approx(-.2)
    a=tmp_path/"forward"; b=tmp_path/"reverse"; a.mkdir(); b.mkdir()
    first,act1=_scene(a,kinds,(False,True),velocity=(.1,-.2))
    second,act2=_scene(b,kinds,(False,True),velocity=(.1,-.2),reverse=True)
    with first.build() as one, second.build() as two:
        out1,out2=one.step(H,actions=act1),two.step(H,actions=act2)
        for body in (10,20):
            np.testing.assert_array_equal(out1.robots[0,body].q,out2.robots[0,body].q)
            np.testing.assert_array_equal(out1.robots[0,body].v,out2.robots[0,body].v)

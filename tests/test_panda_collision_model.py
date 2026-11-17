"""Panda connection geometry and explicit rules, without changing raw imports."""

import xml.etree.ElementTree as ET
from itertools import product

import mujoco
import numpy as np
from test_native_exclusions import eligibility
from test_physolve_models import robot_path

from phymeo.builders.native_assembly import (
    NativeAssetInstance,
    NativeStaticShape,
    compile_native_template,
)
from phymeo.builders.panda import load_panda


def panda_template(*, obstacle=False):
    """Keep zero initial coordinates, Euler, passive dynamics and all other contacts."""
    shapes = (
        (NativeStaticShape("obstacle", "box", (.04, .04, .04), (.06, 0, .16)),)
        if obstacle else ()
    )
    return compile_native_template(
        (NativeAssetInstance("panda", str(robot_path("panda"))),),
        timestep=.002, gravity=(0, 0, -9.81), static_shapes=shapes,
    )


def panda_connection_plan(template):
    """Select the nine source pairs explicitly; no filename-dependent runtime rule."""
    source = template.compilation.collision_source
    owners = ("panda::panda_link0_sc", "panda::panda_link1_sc")
    groups = [[g for g in source.geoms if g.body_name == name] for name in owners]
    assert [len(group) for group in groups] == [3, 3]
    return source.plan_exclusions(
        [(a.key, b.key) for a, b in product(*groups)], source_identity=source.identity,
    )


def test_panda_base_connection_overlap_is_authored_and_rotation_invariant():
    document = ET.parse(robot_path("panda")).getroot()
    joint = document.find("./joint[@name='panda_joint1']")
    np.testing.assert_array_equal(np.fromstring(joint.find("origin").get("xyz"), sep=" "),
                                  [0, 0, .333])
    assert joint.find("axis").get("xyz") == "0 0 1"
    np.testing.assert_array_equal(np.fromstring(joint.find("origin").get("rpy"), sep=" "),
                                  [0, 0, 0])
    shapes = {}
    for body in ("panda_link0_sc", "panda_link1_sc"):
        link = document.find(f"./link[@name='{body}']")
        for shape in link.findall("collision"):
            origin = shape.find("origin")
            position = np.fromstring(origin.get("xyz"), sep=" ")
            geometry = shape.find("geometry")[0]
            if shape.get("name") == "panda_link1_sc_0_barrel":
                np.testing.assert_array_equal(np.fromstring(origin.get("rpy"), sep=" "),
                                              [0, 0, 0])
            shapes[shape.get("name")] = (position, geometry)
    # Independent URDF arithmetic: the link1 cylinder/sphere union is coaxial
    # with joint1. Its world geometric set is invariant under every joint1 angle.
    for suffix in ("barrel", "cap1", "cap2"):
        position, _ = shapes[f"panda_link1_sc_0_{suffix}"]
        np.testing.assert_array_equal(position[:2], [0, 0])
    lower = shapes["panda_link1_sc_0_cap2"][0] + [0, 0, .333]
    barrel, cylinder = shapes["panda_link1_sc_0_barrel"]
    assert cylinder.tag == "cylinder" and float(cylinder.get("radius")) == .06
    zrange = barrel[2] + .333 + np.array([-.5, .5]) * float(cylinder.get("length"))
    expected = {}
    for suffix in ("cap1", "cap2"):
        name = f"panda_link0_sc_0_{suffix}"
        position, sphere = shapes[name]
        assert sphere.tag == "sphere" and float(sphere.get("radius")) == .06
        assert zrange[0] < position[2] < zrange[1]
        expected[name, "panda_link1_sc_0_cap2"] = np.linalg.norm(position - lower) - .12
        expected[name, "panda_link1_sc_0_barrel"] = np.linalg.norm(position[:2]) - .12
    model = mujoco.MjModel.from_xml_path(str(robot_path("panda")))
    for angle in np.linspace(-2.8973, 2.8973, 9):
        data = mujoco.MjData(model)
        data.qpos[0] = angle
        mujoco.mj_forward(model, data)
        found = {}
        for contact in data.contact:
            names = tuple(model.geom(int(g)).name for g in contact.geom)
            if names not in expected:
                continue
            found[names] = contact.dist
            jacobians = []
            for geom in contact.geom:
                linear = np.zeros((3, model.nv))
                angular = np.zeros_like(linear)
                mujoco.mj_jac(model, data, linear, angular, contact.pos,
                             int(model.geom_bodyid[geom]))
                jacobians.append(linear)
            normal = contact.frame[:3] @ (jacobians[1] - jacobians[0])
            np.testing.assert_allclose(normal, 0, atol=1e-12, rtol=0)
        assert found.keys() == expected.keys()
        for names, distance in found.items():
            np.testing.assert_allclose(distance, expected[names], atol=1e-12, rtol=0)


def test_existing_panda_asset_excludes_base_neighbor_but_not_fingers():
    asset = load_panda()
    shapes = asset.collision_model.shapes
    base = [s.key for s in shapes if s.key.body == 0]
    link1 = [s.key for s in shapes if s.key.body == 1]
    assert len(base) == len(link1) == 1  # native three-piece proxies are merged here
    excluded = {frozenset(pair) for pair in asset.collision_model.excluded}
    assert frozenset((base[0], link1[0])) in excluded
    assert not any({a.body, b.body} == {8, 9} for a, b in asset.collision_model.excluded)


def test_panda_explicit_rule_preserves_other_candidates_limits_and_obstacle():
    original = panda_template(obstacle=True)
    plan = panda_connection_plan(original)
    filtered = original.with_collision_exclusions(plan)
    before, after = original.compilation.mj_model, filtered.compilation.mj_model
    assert len(plan.native_geom_pairs) == 9
    assert eligibility(after) == eligibility(before) - set(plan.native_geom_pairs)
    # Apart from exclude/name storage, native arrays and options stay exact.
    for name in dir(before):
        if name.startswith(("_", "name_")) or name in {"exclude_signature", "names", "names_map"}:
            continue
        value = getattr(before, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(after, name), value, err_msg=name)
    for name in dir(before.opt):
        if not name.startswith("_"):
            value = getattr(before.opt, name)
            if isinstance(value, (int, float, np.ndarray)):
                np.testing.assert_array_equal(getattr(after.opt, name), value, err_msg=name)
    for (_, a), (_, b) in zip(original.initial_state.fields, filtered.initial_state.fields):
        np.testing.assert_array_equal(a, b)
    assert not original.collision_plans and before.nexclude == 0

    def contacts(model):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return data, {tuple(sorted(c.geom)) for c in data.contact}

    _, old_pairs = contacts(before)
    data, new_pairs = contacts(after)
    assert new_pairs == old_pairs - set(plan.native_geom_pairs)
    for first, second in (
        ("panda::panda_link5_sc_0_cap1", "panda::panda_hand_sc_0_cap2"),
        ("panda::panda_leftfinger_pad", "panda::panda_rightfinger_pad"),
        ("obstacle", "panda::panda_link1_sc_0_barrel"),
    ):
        assert tuple(sorted((after.geom(first).id, after.geom(second).id))) in new_pairs
    joint4 = after.joint("panda::panda_joint4").id
    np.testing.assert_array_equal(after.jnt_range[joint4], [-3.0718, -.0698])
    assert np.any((data.efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
                  & (data.efc_id == joint4))

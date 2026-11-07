"""Tests-first regression for loss of existing OBB patch material addresses.

UNEXECUTED by the author. These tests sample the frozen F05 scene or exercise
only its record bridge; no contact law, geometry, station gate, or step changes.
"""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from phymeo.constraints.collision_shapes import CollisionKey, CollisionMotion, CollisionShape
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.runtime._robot_scene_contacts import _Records


LAW = RobotContactMaterial(32.0, 0.0, 0.0, 0.01)


def _example():
    name = "phymeo_robot_patch_material_weights_example"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "examples" / "general_scene_friction.py"
        spec = spec_from_file_location(name, path)
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _reconstruct(shape, pose, weights):
    actual_vertices = dict(zip(shape.vertex_ids, np.asarray(pose.start), strict=True))
    return sum((weight * actual_vertices[node] for node, weight in weights), start=np.zeros(3))


def test_f05_real_twelve_sites_keep_declared_faces_and_reconstruct_surface(tmp_path):
    example = _example()
    with example.make_rotating_tool(tmp_path) as case:
        scene = case.scene
        before = example.public_payload(scene.state)
        geometry, loads = scene.geometry(), scene.sample()
        shapes = {shape.key: shape for shape in geometry.model.shapes}
        rotor = scene.shape_key(example.LOWER_TOOL, "tool", "contact")
        support = scene.body_keys[case.world, example.BOX][0]
        declared = {rotor: (0, 2, 4, 6), support: (1, 3, 5, 7)}
        # Independently establish the declared physical faces from actual vertices.
        for key, expected_z in ((rotor, -example.DELTA), (support, 0.0)):
            by_id = dict(zip(shapes[key].vertex_ids, geometry.poses[key].start, strict=True))
            np.testing.assert_allclose([by_id[node][2] for node in declared[key]], expected_z,
                                       atol=1e-12, rtol=0)
        assert sum(load.force_n for load in loads.contacts) == pytest.approx(2.0, abs=1e-9)
        assert sum(len(load.sites) for load in loads.contacts) == 12
        for load in loads.contacts:
            assert {load.first, load.second} == {rotor, support}
            for site in load.sites:
                assert site.force_n > 0
                for side, key in (("first", load.first), ("second", load.second)):
                    weights = getattr(site.material, side + "_weights")
                    point = getattr(site.material, side + "_point")
                    assert all(weight > 0 for _, weight in weights)
                    assert {node for node, _ in weights} <= set(declared[key]), (
                        f"{side} key={key} declared={declared[key]} weights={weights} point={point}"
                    )
                    assert sum(weight for _, weight in weights) == pytest.approx(1, abs=1e-12)
                    np.testing.assert_allclose(
                        _reconstruct(shapes[key], geometry.poses[key], weights), point,
                        atol=1e-12, rtol=0,
                    )
        assert example.public_payload(scene.state) == before
        assert scene.observe().step_number == 0


def _keys(reverse_keys):
    first_body, second_body = (90, 3) if reverse_keys else (3, 90)
    return CollisionKey(7, first_body, 5), CollisionKey(7, second_body, 9)


def _row(pa, pb, ca, cb, *, normal_load, gap):
    return dict(
        normal_world=(0.0, 0.0, 1.0),
        force_on_payload_N=(0.0, 0.0, normal_load),
        link_surface_m=pa,
        payload_surface_m=pb,
        link_com_world_m=ca,
        payload_com_world_m=cb,
        point_world_m=tuple((np.asarray(pa) + pb) / 2),
        normal_load_N=normal_load,
        normal_rate_m_s=0.0,
        weight=1.0,
        pair_power_W=0.0,
        gap_m=gap,
        feature="face",
    )


def _check_record(records, keys, points, expected_weights, row):
    assert len(records.contacts) == len(records.details) == 1
    load, detail = records.contacts[0], records.details[0]
    assert (load.first, load.second) == tuple(sorted(keys))
    site = load.sites[0]
    sign = 1 if load.first == keys[0] else -1
    np.testing.assert_array_equal(load.normal, sign * np.array([0, 0, 1]))
    np.testing.assert_array_equal(site.force_on_second_n, sign * np.asarray(row["force_on_payload_N"]))
    assert site.force_n == row["normal_load_N"] and load.gap_m == row["gap_m"]
    assert site.application_point_first_m == site.application_point_second_m == row["point_world_m"]
    for side, key in (("first", load.first), ("second", load.second)):
        actual = getattr(site.material, side + "_weights")
        assert actual == expected_weights[key]
        assert detail[side + "_weights"] == expected_weights[key]
        np.testing.assert_array_equal(getattr(site.material, side + "_point"), points[key])
        np.testing.assert_array_equal(detail[side + "_surface_m"], points[key])


@pytest.mark.parametrize("reverse_keys", [False, True], ids=["ordered", "swapped"])
@pytest.mark.parametrize("explicit", [False, True], ids=["metadata", "explicit-precedence"])
def test_records_keep_local_patch_chart_ids_and_explicit_precedence(reverse_keys, explicit):
    first, second = _keys(reverse_keys)
    a = CollisionShape(first, "hull", (101, 305, 701, 901, 1001, 1305, 1701, 1901))
    b = CollisionShape(second, "hull", (7, 13, 23, 31, 47, 61, 79, 97))
    corners = np.array(list(product((-1.0, 1.0), repeat=3)))
    poses = {first: CollisionMotion(corners),
             second: CollisionMotion(corners + [0, 0, 15 / 8])}
    records = _Records({first: a, second: b}, poses)
    points = {first: (0.0, 0.0, 1.0), second: (0.0, 0.0, 7 / 8)}
    row = _row(points[first], points[second], (0, 0, 0), (0, 0, 15 / 8),
               normal_load=4.0, gap=-1 / 8)
    # Two diagonal endpoints give each true face centre. A fresh trilinear fit
    # would use four vertices, so equality proves retention of the original chart.
    row["link_material_weights"] = ((1, .5), (7, .5))
    row["table_material_weights"] = ((2, .5), (4, .5))
    expected = {first: ((305, .5), (1901, .5)), second: ((23, .5), (47, .5))}
    if explicit:
        # The alternate face diagonals reconstruct the same points. Existing
        # explicit addresses must win without treating material IDs as indices.
        row["first_weights"] = ((901, .5), (1305, .5))
        row["second_weights"] = ((7, .5), (79, .5))
        expected = {first: row["first_weights"], second: row["second_weights"]}
    records.add(first, second, row, LAW, "robot-structure")
    _check_record(records, (first, second), points, expected, row)
    for key, shape in ((first, a), (second, b)):
        np.testing.assert_array_equal(_reconstruct(shape, poses[key], expected[key]), points[key])


@pytest.mark.parametrize("reverse_keys", [False, True], ids=["ordered", "swapped"])
def test_records_without_metadata_keep_sphere_capsule_material_fallback(reverse_keys):
    first, second = _keys(reverse_keys)
    sphere = CollisionShape(first, "sphere", (71,), radius=1 / 8)
    capsule = CollisionShape(second, "capsule", (43, 101), radius=1 / 8)
    poses = {
        first: CollisionMotion([[.5, 0, 0]]),
        second: CollisionMotion([[0, 0, 3 / 16], [2, 0, 3 / 16]]),
    }
    records = _Records({first: sphere, second: capsule}, poses)
    points = {first: (.5, 0, 1 / 8), second: (.5, 0, 1 / 16)}
    row = _row(points[first], points[second], (.5, 0, 0), (1, 0, 3 / 16),
               normal_load=2.0, gap=-1 / 16)
    assert not any(name in row for name in (
        "first_weights", "second_weights", "link_material_weights", "table_material_weights",
    ))
    records.add(first, second, row, LAW, "robot-structure")
    expected = {first: ((71, 1.0),), second: ((43, .75), (101, .25))}
    _check_record(records, (first, second), points, expected, row)
    # Rounded material addresses name the centre/spine; the recorded physical
    # surface carries the original radius offset, rather than a false hull point.
    np.testing.assert_array_equal(_reconstruct(sphere, poses[first], expected[first]), [.5, 0, 0])
    np.testing.assert_array_equal(_reconstruct(capsule, poses[second], expected[second]), [.5, 0, 3 / 16])

"""Finite body-local box-union sides; no physics integration or GPU needed."""

import mujoco as mj
import numpy as np
import pytest

from experiments.area_pressure import pusht_public_scene as public
from phymeo.backends.mujoco_heterogeneous_area_scene import _asset_digest
from phymeo.geometry.area_pressure_intersection import RigidPressureMesh
from phymeo.training.manipulation_tasks import manipulation_scenes

LATERAL = ((0, -1), (0, 1), (1, -1), (1, 1))
LEGACY_SHA = "b95fb5d94ba4ad7cbbdc078e954482200969ac160867a616d5052bd7e238172e"


@pytest.fixture(scope="module")
def original():
    return next(scene for scene in manipulation_scenes() if scene.key == "pusht-planar")


def _mesh(bounds, directions=LATERAL):
    from phymeo.geometry.box_union_surface import axis_aligned_box_union_mesh

    vertices, faces = axis_aligned_box_union_mesh(bounds, face_directions=directions)
    return RigidPressureMesh(
        tuple(f"v{i}" for i in range(len(vertices))),
        vertices,
        faces,
        tuple(f"f{i}" for i in range(len(faces))),
    )


def _vectors(mesh):
    triangle = mesh.world_vertices[mesh.faces]
    return np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])


def _assert_exterior(mesh, boxes):
    vectors = _vectors(mesh)
    assert np.all(np.count_nonzero(vectors, axis=1) == 1)
    assert np.all(vectors[:, 2] == 0)  # Deliberately no top or bottom.
    normal = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    center = mesh.world_vertices[mesh.faces].mean(axis=1)
    for point, outward in zip(center, normal, strict=True):
        inside = point - 1e-6 * outward
        outside = point + 1e-6 * outward

        def contains(p):
            return np.any(np.all((boxes[:, 0] < p) & (p < boxes[:, 1]), axis=1))

        assert contains(inside)
        assert not contains(outside)
    assert len(
        {tuple(sorted(map(tuple, triangle))) for triangle in mesh.world_vertices[mesh.faces]}
    ) == len(mesh.faces)


@pytest.mark.parametrize(
    "boxes, expected_area",
    (
        ([((-2, -1, -0.5), (2, 1, 0.5))], 12.0),
        ([((-2, -1, -0.5), (1, 1, 0.5)), ((-1, -1, -0.5), (2, 1, 0.5))], 12.0),
        ([((-1, -3, -1), (1, 1, 1)), ((-3, 1, -1), (3, 3, 1))], 48.0),
    ),
)
def test_union_sides_are_complete_outward_and_have_no_internal_or_duplicate_faces(
    boxes, expected_area
):
    boxes = np.asarray(boxes, dtype=float)
    mesh = _mesh(boxes)
    _assert_exterior(mesh, boxes)
    assert np.linalg.norm(_vectors(mesh), axis=1).sum() / 2 == expected_area
    assert set(map(tuple, np.sign(_vectors(mesh)))) == {
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
    }


def test_union_is_invariant_to_box_order_and_duplicate_solids():
    boxes = np.array([((-1, -3, -1), (1, 1, 1)), ((-3, 1, -1), (3, 3, 1))], dtype=float)
    first, second = _mesh(boxes), _mesh(boxes[[1, 0, 1]])
    np.testing.assert_array_equal(first.world_vertices, second.world_vertices)
    np.testing.assert_array_equal(first.faces, second.faces)


def test_thin_real_gap_is_not_filled_by_midpoint_rounding_or_welding():
    right = np.nextafter(1.0, np.inf)
    boxes = np.array([((0, 0, 0), (1, 1, 1)), ((right, 0, 0), (2, 1, 1))])
    mesh = _mesh(boxes)
    triangles = mesh.world_vertices[mesh.faces]
    vectors = _vectors(mesh)
    assert np.count_nonzero(np.all(triangles[:, :, 0] == 1.0, axis=1) & (vectors[:, 0] > 0)) == 2
    assert np.count_nonzero(np.all(triangles[:, :, 0] == right, axis=1) & (vectors[:, 0] < 0)) == 2


@pytest.mark.parametrize(
    "bad",
    (
        [],
        [[[0, 0], [1, 1]]],
        [[[0, 0, 0], [1, 1, np.inf]]],
        [[[0, 0, 0], [1, 1, 0]]],
        [[[1, 0, 0], [0, 1, 1]]],
        [[[False, False, False], [True, True, True]]],
    ),
)
def test_invalid_box_bounds_fail_explicitly(bad):
    with pytest.raises(ValueError, match="bounds"):
        _mesh(bad)


@pytest.mark.parametrize("bad", ((), ((2, 0),), ((3, 1),), ((0, True),), ((1, -1), (1, -1))))
def test_invalid_face_direction_catalog_fails_explicitly(bad):
    with pytest.raises(ValueError, match="face_directions"):
        _mesh([((0, 0, 0), (1, 1, 1))], bad)


def test_legacy_default_material_identity_order_and_asset_hash_are_unchanged(original):
    model = mj.MjModel.from_xml_string(original.mjcf)
    implicit = public.build_side_asset(model)
    # This literal is the accepted pre-extension scene asset SHA, independent
    # of the new implementation and requiring no ignored archive at test time.
    assert _asset_digest({"original-RR-public-mixed-area": (implicit,)}) == LEGACY_SHA
    explicit = public.build_side_asset(model, side_coverage="negative-y")
    assert _asset_digest({"original-RR-public-mixed-area": (explicit,)}) == LEGACY_SHA
    assert implicit.identity == explicit.identity == public.IDENTITY
    assert public.side_identity() == public.IDENTITY
    np.testing.assert_array_equal(implicit.rigid_local.faces, explicit.rigid_local.faces)
    np.testing.assert_array_equal(
        implicit.rigid_local.world_vertices, explicit.rigid_local.world_vertices
    )
    assert implicit.rigid_local.face_ids == explicit.rigid_local.face_ids


def test_all_lateral_keeps_pressure_material_and_binds_distinct_complete_scene_law(original):
    model = mj.MjModel.from_xml_string(original.mjcf)
    old = public.build_side_asset(model)
    asset = public.build_side_asset(model, side_coverage="all-lateral")
    spec = public.build_scene_spec(original.mjcf, side_coverage="all-lateral")
    assert asset.identity == public.side_identity("all-lateral") == public.ALL_LATERAL_IDENTITY
    assert asset.identity != old.identity
    assert asset.identity.compliant_material == old.identity.compliant_material
    assert asset.identity.rigid_material == "original-T-finite-all-lateral-faces"
    assert spec.area_pressure_pairs == (("tool", "object"),)
    (law,) = spec.area_friction_laws
    assert (
        law.identity == asset.identity and law.mu == public.MU and law.slip_speed_m_s == public.SLIP
    )
    for name in ("world_vertices", "cells", "nodal_pressure"):
        np.testing.assert_array_equal(
            getattr(old.pressure_local, name), getattr(asset.pressure_local, name)
        )
    assert len(asset.rigid_local.faces) > len(old.rigid_local.faces)
    assert np.all(_vectors(asset.rigid_local)[:, 2] == 0)
    assert _asset_digest({spec.key: (asset,)}) != LEGACY_SHA


def test_body_pose_does_not_rotate_or_translate_body_local_material(original):
    model = mj.MjModel.from_xml_string(original.mjcf)
    first = public.build_side_asset(model, side_coverage="all-lateral")
    body = model.body("object").id
    model.body_pos[body] += (3.0, -2.0, 4.0)
    model.body_quat[body] = (np.cos(0.2), 0, 0, np.sin(0.2))
    second = public.build_side_asset(model, side_coverage="all-lateral")
    np.testing.assert_array_equal(
        first.rigid_local.world_vertices, second.rigid_local.world_vertices
    )
    np.testing.assert_array_equal(first.rigid_local.faces, second.rigid_local.faces)


@pytest.mark.parametrize("fault", ("box-rotation", "tool-rotation", "box-owner", "box-type"))
def test_unsupported_local_geometry_is_rejected_instead_of_silently_axis_aligning(original, fault):
    model = mj.MjModel.from_xml_string(original.mjcf)
    box, tool = model.geom("object-0").id, model.geom("pusher").id
    if fault.endswith("rotation"):
        model.geom_quat[box if fault == "box-rotation" else tool] = (np.cos(0.2), 0, 0, np.sin(0.2))
    elif fault == "box-owner":
        model.geom_bodyid[box] = model.body("tool").id
    else:
        model.geom_type[box] = int(mj.mjtGeom.mjGEOM_SPHERE)
    with pytest.raises(ValueError, match="body-local|axis-aligned|capsule|box"):
        public.build_side_asset(model, side_coverage="all-lateral")


@pytest.mark.parametrize("bad", (None, "all", "positive-y", 3))
def test_unknown_coverage_is_not_silently_defaulted(original, bad):
    model = mj.MjModel.from_xml_string(original.mjcf)
    with pytest.raises(ValueError, match="side_coverage"):
        public.build_side_asset(model, side_coverage=bad)
    with pytest.raises(ValueError, match="side_coverage"):
        public.build_scene_spec(original.mjcf, side_coverage=bad)


def test_declared_decimal_material_join_survives_actual_world_transform(original):
    from dataclasses import replace

    model = mj.MjModel.from_xml_string(original.mjcf)
    asset = public.build_side_asset(model, side_coverage="all-lateral")
    data = mj.MjData(model)
    # Accepted original A ordinary-route input 4450, fixed portable q only.
    # FK reconstructs the saved pose; this does not advance any physics.
    data.qpos[:] = (
        -0.6462836861610413,
        0.9709684252738953,
        0.5205203890800476,
        0.01619715243577957,
        0.014954738318920135,
        0.9984166026115417,
        -1.801822264724251e-7,
        1.3197508863527219e-8,
        -0.05625271424651146,
    )
    mj.mj_kinematics(model, data)
    body = model.body("object").id
    world = asset.rigid_local.world_vertices @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
    assert len({tuple(v) for v in world}) == len(world)
    replace(asset.rigid_local, world_vertices=world)  # Original geometry guard.


def test_named_material_coordinate_policy_records_shared_join_and_every_delta(original):
    model = mj.MjModel.from_xml_string(original.mjcf)
    evidence = public.side_geometry_provenance(model, side_coverage="all-lateral")
    assert evidence["coordinate_policy"] == "shortest-decimal-center-halfsize-single-round-v1"
    assert evidence["coordinate_frame"] == "object-body-local"
    assert (
        evidence["source_interpretation"]
        == "compiled-float-shortest-roundtrip-decimal-not-original-XML"
    )
    boxes = {box["geometry_name"]: box for box in evidence["boxes"]}
    stem, bar = boxes["object-0"], boxes["object-1"]
    a, b = (
        np.asarray(stem["material_bounds_body_local_m"]),
        np.asarray(bar["material_bounds_body_local_m"]),
    )
    assert a[1, 1] == b[0, 1] == 0.02
    assert a[0, 1] == -0.07 and b[1, 1] == 0.056
    assert tuple(a[:, 0]) == (-0.018, 0.018) and tuple(b[:, 0]) == (-0.065, 0.065)
    for box in boxes.values():
        geom = model.geom(box["geometry_name"]).id
        raw = np.array(
            (
                model.geom_pos[geom] - model.geom_size[geom],
                model.geom_pos[geom] + model.geom_size[geom],
            )
        )
        np.testing.assert_array_equal(box["raw_bounds_body_local_m"], raw)
        material = np.asarray(box["material_bounds_body_local_m"])
        np.testing.assert_array_equal(box["material_minus_raw_m"], material - raw)
        assert box["center_decimal_m"] == [str(float(x)) for x in model.geom_pos[geom]]
        assert box["half_size_decimal_m"] == [str(float(x)) for x in model.geom_size[geom]]
    assert stem["raw_bounds_body_local_m"][1][1] != bar["raw_bounds_body_local_m"][0][1]
    asset = public.build_side_asset(model, side_coverage="all-lateral")
    # No tiny midpoint strip survives anywhere in the declared local material.
    assert set(asset.rigid_local.world_vertices[:, 1]) == {-0.07, 0.02, 0.056}


def test_explicit_distinct_decimal_coordinates_remain_a_real_material_gap(original):
    model = mj.MjModel.from_xml_string(original.mjcf)
    model.geom_pos[model.geom("object-1").id, 1] = 0.03800000000000001
    evidence = public.side_geometry_provenance(model, side_coverage="all-lateral")
    boxes = {box["geometry_name"]: box for box in evidence["boxes"]}
    lower = boxes["object-1"]["material_bounds_body_local_m"][0][1]
    upper = boxes["object-0"]["material_bounds_body_local_m"][1][1]
    assert lower > upper == 0.02
    asset = public.build_side_asset(model, side_coverage="all-lateral")
    assert lower in asset.rigid_local.world_vertices[:, 1]
    assert upper in asset.rigid_local.world_vertices[:, 1]

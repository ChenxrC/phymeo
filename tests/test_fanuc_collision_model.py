"""FANUC asset/import audit; pair-filter controls are not a production policy.

The raw asset and original runtime qualification tests remain unchanged. These
tests use the authored URDF, never a Newton-exported reference model.
"""

import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from phymeo.builders.mujoco_compiler import MuJoCoCompilation, compile_mujoco
from test_physolve_models import robot_path
from test_physolve_runtime import payload as payload


def stl_triangles(path):
    content = path.read_bytes()
    count = struct.unpack_from("<I", content, 80)[0]
    assert len(content) == 84 + 50 * count, "audit requires binary STL"
    return np.ndarray((count, 3, 3), dtype="<f4", buffer=content,
                      offset=96, strides=(50, 12, 4)).astype(np.float64)


def source_meshes():
    path = robot_path("fanuc")
    document = ET.parse(path).getroot()
    joint = document.find("./joint[@name='joint_1']")
    assert joint.find("parent").get("link") == "base_link"
    assert joint.find("child").get("link") == "link_1"
    # This audit deliberately asserts the authored zero rotations instead of
    # borrowing MuJoCo's transform or silently handling an unrelated asset.
    np.testing.assert_array_equal(np.fromstring(joint.find("origin").get("rpy"), sep=" "), [0, 0, 0])
    offset = np.fromstring(joint.find("origin").get("xyz"), sep=" ")
    result = {}
    for name, body_position in (("base_link", np.zeros(3)), ("link_1", offset)):
        link = document.find(f"./link[@name='{name}']")
        for kind in ("collision", "visual"):
            element = link.find(kind)
            origin = element.find("origin")
            np.testing.assert_array_equal(np.fromstring(origin.get("rpy"), sep=" "), [0, 0, 0])
            mesh = element.find("geometry/mesh")
            assert mesh.get("scale") is None
            points = stl_triangles(path.parent / mesh.get("filename"))
            result[name, kind] = points + body_position + np.fromstring(origin.get("xyz"), sep=" ")
    return result


def winding_number(triangles, point):
    vectors = triangles - point
    a, b, c = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    lengths = np.linalg.norm(vectors, axis=2)
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = (lengths.prod(axis=1)
                   + np.einsum("ij,ij->i", a, b) * lengths[:, 2]
                   + np.einsum("ij,ij->i", b, c) * lengths[:, 0]
                   + np.einsum("ij,ij->i", c, a) * lengths[:, 1])
    return np.arctan2(numerator, denominator).sum() / (2 * np.pi)


def pair_geoms(model):
    import mujoco
    result = []
    for name in ("base_link", "link_1"):
        mesh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, name)
        assert mesh >= 0
        geoms = np.flatnonzero((model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
                              & (model.geom_dataid == mesh))
        assert len(geoms) == 1
        result.append(int(geoms[0]))
    return result


def filter_only_base_link1(model):
    """Diagnostic masks: remove exactly one pair; retain all other mask pairs."""
    base, link = pair_geoms(model)
    np.testing.assert_array_equal(model.geom_contype, np.ones(model.ngeom))
    np.testing.assert_array_equal(model.geom_conaffinity, np.ones(model.ngeom))
    before = ((model.geom_contype[:, None] & model.geom_conaffinity[None, :])
              | (model.geom_contype[None, :] & model.geom_conaffinity[:, None])) != 0
    model.geom_conaffinity[:] = 7
    model.geom_contype[base], model.geom_contype[link] = 2, 4
    model.geom_conaffinity[base] = model.geom_conaffinity[link] = 1
    after = ((model.geom_contype[:, None] & model.geom_conaffinity[None, :])
             | (model.geom_contype[None, :] & model.geom_conaffinity[:, None])) != 0
    expected = before.copy()
    expected[base, link] = expected[link, base] = False
    # Same-geom pairs are not collision candidates.
    np.fill_diagonal(after, False)
    np.fill_diagonal(expected, False)
    np.testing.assert_array_equal(after, expected)


def test_raw_collision_meshes_overlap_but_visuals_have_vertical_clearance():
    meshes = source_meshes()
    points = np.array([[0., 0., .30], [.05, .04, .30], [.10, .05, .30]])
    for name in ("base_link", "link_1"):
        triangles = meshes[name, "collision"]
        _, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3)
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, np.full(len(counts), 2))
        np.testing.assert_allclose([winding_number(triangles, p) for p in points], 1, atol=1e-6, rtol=0)
        np.testing.assert_allclose(winding_number(triangles, np.array([2., 2., 2.])), 0, atol=1e-6, rtol=0)
    overlap = meshes["base_link", "collision"][:, :, 2].max() - meshes["link_1", "collision"][:, :, 2].min()
    gap = meshes["link_1", "visual"][:, :, 2].min() - meshes["base_link", "visual"][:, :, 2].max()
    np.testing.assert_allclose(overlap, .067, atol=1e-6, rtol=0)
    assert gap > 0, "visuals are disjoint along z; this is not a CAD certification"


def test_native_and_phymeo_preserve_authored_mesh_coordinates_and_contacts():
    import mujoco
    meshes = source_meshes()
    record = compile_mujoco(robot_path("fanuc"), require_asset=False)
    for model in (mujoco.MjModel.from_xml_path(str(robot_path("fanuc"))), record.mj_model, record.fresh_native_model()):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        base, link = pair_geoms(model)
        assert model.geom_bodyid[base] == 0
        assert model.body_parentid[model.geom_bodyid[link]] == 0
        assert model.nexclude == 0
        for name, geom in zip(("base_link", "link_1"), (base, link)):
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            actual = model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            expected = np.unique(meshes[name, "collision"].reshape(-1, 3), axis=0)
            assert len(actual) == len(expected)
            distances = np.max(np.abs(actual[:, None] - expected[None, :]), axis=2)
            assert distances.min(axis=0).max() < 1e-6
            assert distances.min(axis=1).max() < 1e-6
        contacts = [c for c in data.contact if set(c.geom) == {base, link}]
        assert len(contacts) == 4
        np.testing.assert_allclose([c.dist for c in contacts], -.067, atol=1e-6, rtol=0)


def test_pair_filter_preserves_world_obstacle_contacts():
    import mujoco
    spec = mujoco.MjSpec.from_file(str(robot_path("fanuc")))
    spec.worldbody.add_geom(name="audit_plane", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[2, 2, .01], pos=[0, 0, .28])
    model = spec.compile()
    base, link = pair_geoms(model)
    plane = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "audit_plane")
    def pairs():
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return {tuple(sorted(c.geom)) for c in data.contact}
    before = pairs()
    filter_only_base_link1(model)
    after = pairs()
    assert tuple(sorted((base, link))) in before
    assert tuple(sorted((base, link))) not in after
    assert tuple(sorted((plane, link))) in before & after
    assert after == before - {tuple(sorted((base, link)))}


def test_base_link1_penetration_cannot_be_resolved_by_joint_motion():
    import mujoco
    model = compile_mujoco(robot_path("fanuc"), require_asset=False).fresh_native_model()
    base, link = pair_geoms(model)
    for angle in np.linspace(*model.jnt_range[0], 9):
        data = mujoco.MjData(model)
        data.qpos[0] = angle
        mujoco.mj_forward(model, data)
        contacts = [c for c in data.contact if set(c.geom) == {base, link}]
        assert len(contacts) == 4
        np.testing.assert_allclose([c.dist for c in contacts], -.067, atol=1e-6, rtol=0)
        for contact in contacts:
            jacobians = []
            for geom in contact.geom:
                translation = np.zeros((3, model.nv))
                rotation = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, translation, rotation, contact.pos, int(model.geom_bodyid[geom]))
                jacobians.append(translation)
            normal = contact.frame.reshape(3, 3)[0]
            # The normal separation has no available generalized velocity.
            np.testing.assert_allclose(normal @ (jacobians[1] - jacobians[0]), 0, atol=1e-6, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_pair_filtered_diagnostic_keeps_existing_motion_forward_reset_checks(payload, monkeypatch, nworld):
    from phymeo import PhySolveRuntime
    from test_physolve_runtime import test_robot_forward_and_masked_reset_native_state
    original = MuJoCoCompilation.fresh_native_model
    snapshots = []
    snapshot = PhySolveRuntime.snapshot
    def capture(runtime):
        value = snapshot(runtime)
        snapshots.append(value)
        return value
    def diagnostic(record):
        model = original(record)
        filter_only_base_link1(model)
        return model
    monkeypatch.setattr(MuJoCoCompilation, "fresh_native_model", diagnostic)
    monkeypatch.setattr(PhySolveRuntime, "snapshot", capture)
    # Reuse the original assertions, including exact reset and 0.03 N/Nm action.
    # This separate diagnostic profile does NOT change the failing raw test.
    test_robot_forward_and_masked_reset_native_state(payload, "fanuc", nworld)
    initial, advanced = snapshots[:2]
    assert np.all(np.any(advanced["qvel"] != initial["qvel"], axis=1))
    for field in ("qpos", "qvel"):
        np.testing.assert_allclose(advanced[field], np.broadcast_to(advanced[field][0], advanced[field].shape),
                                   atol=1e-6, rtol=0)
    print("FANUC pair-filter diagnostic", nworld, "qvel[0]", advanced["qvel"][0].tolist())

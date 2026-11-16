"""Cold visual identity/export, independent from unchanged physical geometry."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from test_native_assembly import URDF

from phymeo.builders.native_assembly import NativeAssetInstance, compile_native_template


def build(path, capture=True):
    return compile_native_template(
        (NativeAssetInstance("arm", str(path)),),
        timestep=0.002,
        gravity=(0, 0, 0),
        capture_visuals=capture,
    )


def test_urdf_visual_does_not_change_physics_and_survives_source_removal(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text(
        URDF.replace(
            "<collision>",
            '<visual><origin xyz=".1 .2 .3"/><geometry><sphere radius=".4"/></geometry><material name="red"><color rgba="1 0 0 1"/></material></visual><collision>',
        )
    )
    before, after = build(path, False), build(path)
    a, b = before.compilation.fresh_native_model(), after.compilation.fresh_native_model()
    for name in dir(a):
        if isinstance(getattr(a, name), np.ndarray):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    for (name, a), (other, b) in zip(before.initial_state.fields, after.initial_state.fields):
        assert name == other
        np.testing.assert_array_equal(a, b)
    path.unlink()
    scene = after.compile_visual_scene()
    assert len(scene.instances) == 1
    item = scene.instances[0]
    assert (item.body, item.kind, item.source_role) == ("arm::link", "sphere", "visual")
    assert item.position == (0.1, 0.2, 0.3) and item.size == (0.4,) and item.rgba == (1, 0, 0, 1)
    with pytest.raises(RuntimeError, match="capture_visuals"):
        before.compile_visual_scene()


def test_collision_only_fallback_and_native_world_binding(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text(URDF)
    scene = build(path).compile_visual_scene()
    assert scene.instances[0].source_role == "collision_geometry"
    path = tmp_path / "native.xml"
    path.write_text(
        '<mujoco><worldbody><geom name="floor" type="box" size="1 1 .1" pos="0 0 -1"/><body name="b"><geom name="ball" type="sphere" size=".2" rgba="0 0 1 1"/></body></worldbody></mujoco>'
    )
    scene = build(path).compile_visual_scene()
    floor = next(i for i in scene.instances if i.name == "arm::floor")
    assert floor.body is None and floor.position == (0, 0, -1)
    assert all(i.source_role == "native_geom" for i in scene.instances)


def test_mesh_relative_resources_and_immutable_buffers(tmp_path):
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
    path = tmp_path / "robot.urdf"
    path.write_text(
        URDF.replace(
            "<collision>",
            '<visual><geometry><mesh filename="mesh.obj" scale=".2 .3 .4"/></geometry></visual><collision>',
        )
    )
    template = build(path)
    mesh.unlink()
    path.unlink()
    scene = template.compile_visual_scene()
    assert scene.instances[0].kind == "mesh"
    assert len(scene.meshes) == 1
    vertices = scene.meshes[0].vertices
    assert vertices.shape == (4, 3)
    with pytest.raises(ValueError):
        vertices.setflags(write=True)
    with pytest.raises(ValueError):
        scene.meshes[0].triangles[0, 0] = 10


def test_visual_export_does_not_recompile_or_reopen(tmp_path, monkeypatch):
    path = tmp_path / "robot.urdf"
    path.write_text(URDF)
    template = build(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("late source access")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert template.compile_visual_scene() is template.compile_visual_scene()


def test_finite_plane_rectangle_and_group_visibility(tmp_path):
    path = tmp_path / "plane.xml"
    path.write_text(
        '<mujoco><worldbody><geom name="plane" type="plane" size="2 3 .1" pos="1 2 3" euler="0 0 90"/><geom name="hidden" type="sphere" size="1" group="3"/></worldbody></mujoco>'
    )
    scene = build(path).compile_visual_scene()
    plane = next(i for i in scene.instances if i.name == "arm::plane")
    assert plane.kind == "mesh" and plane.body is None
    mesh = next(m for m in scene.meshes if m.name == plane.mesh)
    np.testing.assert_array_equal(mesh.vertices.min(axis=0), [-2, -3, 0])
    np.testing.assert_array_equal(mesh.vertices.max(axis=0), [2, 3, 0])
    np.testing.assert_allclose(plane.rotation_xyzw, [0, 0, 2**-0.5, 2**-0.5], atol=1e-12)
    assert not next(i for i in scene.instances if i.name == "arm::hidden").visible


def test_texture_failure_does_not_change_physical_compilation(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text(
        URDF.replace(
            "<collision>",
            '<visual><geometry><sphere radius=".4"/></geometry><material name="t"><texture filename="unsupported.png"/></material></visual><collision>',
        )
    )
    ordinary, display = build(path, False), build(path)
    np.testing.assert_array_equal(
        ordinary.compilation.fresh_native_model().geom_type,
        display.compilation.fresh_native_model().geom_type,
    )
    with pytest.raises(NotImplementedError, match="texture"):
        display.compile_visual_scene()


def test_uv_corner_indices_are_exported_in_vertex_space(tmp_path):
    mesh = tmp_path / "uv.obj"
    mesh.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nvt 0 0\nvt 1 0\nvt 0 1\nvt .5 .5\nf 1/1 3/3 2/2\nf 1/4 2/2 4/3\nf 1/1 4/4 3/3\nf 2/2 3/3 4/4\n"
    )
    path = tmp_path / "robot.urdf"
    path.write_text(
        URDF.replace(
            "<collision>",
            '<visual><geometry><mesh filename="uv.obj"/></geometry></visual><collision>',
        )
    )
    scene = build(path).compile_visual_scene()
    output = scene.meshes[0]
    assert output.vertices.shape == (12, 3) and output.uvs.shape == (12, 2)
    np.testing.assert_array_equal(output.triangles, np.arange(12).reshape(4, 3))
    # The public snapshot preserves compiled native UVs; the OBJ importer flips V.
    assert {tuple(row) for row in output.uvs} == {(0, 0), (1, 1), (0, 1), (0.5, 0.5)}


@pytest.mark.parametrize("robot", ["panda", "fanuc"])
def test_real_robot_visual_identity_and_unchanged_physics(robot):
    import phymeo.builders

    if robot == "panda":
        path = Path(phymeo.builders.__file__).parent / "panda_assets/panda.urdf"
    else:
        config = os.environ.get("PHYMEO_PHYSOLVE_SOURCES")
        if not config:
            pytest.skip("explicit FANUC source configuration required")
        path = Path(
            json.loads(Path(config).read_text(encoding="utf-8-sig"))["robot_assets"]["fanuc"]
        )
    ordinary, captured = build(path, False), build(path)
    a, b = ordinary.compilation.fresh_native_model(), captured.compilation.fresh_native_model()
    for name in dir(a):
        if isinstance(getattr(a, name), np.ndarray):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    scene = captured.compile_visual_scene()
    names = {name for name, _ in captured.body_bindings}
    assert scene.instances
    assert all(i.body is None or i.body in names for i in scene.instances)
    if robot == "fanuc":
        assert scene.meshes and all(i.source_role == "visual" for i in scene.instances)
    else:
        assert all(i.source_role == "collision_geometry" for i in scene.instances)

"""Cold rule recompilation owns directory-qualified and cube resources."""

import struct
import zlib

import mujoco
import numpy as np
import pytest

from phymeo.builders.native_assembly import NativeAssetInstance, compile_native_template


def _ascii_tetra():
    points = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    lines = ["solid tetra"]
    for face in faces:
        lines.extend(["facet normal 0 0 0", "outer loop"])
        lines.extend("vertex " + " ".join(map(str, points[i])) for i in face)
        lines.extend(["endloop", "endfacet"])
    lines.append("endsolid tetra")
    return "\n".join(lines).encode("ascii"), points, faces


def test_ascii_stl_encoding_and_solid_binary_header():
    from phymeo.builders._native_stl import binary_stl

    source, points, faces = _ascii_tetra()
    result = binary_stl(source)
    assert len(result) == 284 and struct.unpack_from("<I", result, 80)[0] == 4
    for i, face in enumerate(faces):
        row = struct.unpack_from("<12fH", result, 84 + 50 * i)
        assert row == (0., 0., 0., *(v for j in face for v in points[j]), 0)
    binary = b"solid binary".ljust(80, b"\0") + result[80:]
    assert binary_stl(binary) is binary


@pytest.mark.parametrize("old,new", [(b"endfacet", b"badfacet"),
    (b"vertex 0", b"vertex nan"), (b"vertex 0", b"vertex 1e40"),
    (b"endsolid tetra", b""), (b"outer loop", b"outer")])
def test_bad_ascii_stl_is_not_repaired(old, new):
    from phymeo.builders._native_stl import binary_stl

    with pytest.raises(ValueError, match="ASCII STL"):
        binary_stl(_ascii_tetra()[0].replace(old, new))


def test_ascii_stl_physics_and_visuals_frozen_without_source_mutation(tmp_path):
    from phymeo.builders._native_stl import binary_stl

    source = _ascii_tetra()[0]
    mesh = tmp_path / "tetra.stl"
    mesh.write_bytes(source)
    path = tmp_path / "robot.urdf"
    path.write_text('<robot name="r"><link name="base"><inertial><mass value="1"/>'
        '<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>'
        '<collision><geometry><mesh filename="tetra.stl"/></geometry></collision>'
        '<visual><geometry><mesh filename="tetra.stl"/></geometry></visual></link></robot>')
    def build():
        return compile_native_template((NativeAssetInstance("arm", str(path)),),
            timestep=.002, gravity=(0, 0, 0), capture_visuals=True)
    ascii_model = build()
    assert mesh.read_bytes() == source
    mesh.write_bytes(binary_stl(source))
    binary_model = build()
    a, b = ascii_model.compilation.mj_model, binary_model.compilation.mj_model
    for name in ("mesh_vert", "mesh_face", "body_mass", "body_inertia", "geom_pos"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    mesh.unlink()
    path.unlink()
    visual = ascii_model.compile_visual_scene()
    assert len(visual.meshes) == 1 and len(visual.instances) == 1
    np.testing.assert_array_equal(visual.meshes[0].vertices,
        binary_model.compile_visual_scene().meshes[0].vertices)
    assert ascii_model.compilation.fresh_native_model().nmesh == 1


def _png(values):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    pixels = np.asarray(values, dtype=np.uint8).reshape(2, 2, 3)
    data = b"".join(b"\0" + row.tobytes() for row in pixels)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(data))
        + chunk(b"IEND", b"")
    )


@pytest.mark.parametrize("strippath", [False, True])
def test_hfield_directory_same_filename_owned_and_cube_faces_rejected(tmp_path, strippath):
    heights, images = tmp_path / "heights", tmp_path / "images"
    heights.mkdir()
    images.mkdir()
    (heights / "shared.png").write_bytes(_png(np.repeat([0, 64, 128, 255], 3)))
    (images / "shared.png").write_bytes(_png(np.tile([127, 32, 64], 4)))
    faces = ("right", "left", "up", "down", "front", "back")
    for i, face in enumerate(faces):
        (images / f"{face}.png").write_bytes(_png(np.tile([i * 30, 100, 200], 4)))
    cube = " ".join(f'file{face}="{face}.png"' for face in faces)
    path = tmp_path / "model.xml"
    path.write_text(
        '<mujoco><compiler meshdir="heights" texturedir="images"/><asset>'
        '<hfield name="terrain" size="1 1 .2 .1" file="shared.png"/>'
        '<texture name="surface" type="2d" file="shared.png"/>'
        f'<texture name="cube" type="cube" {cube}/>'
        '<material name="material" texture="surface"/></asset><worldbody>'
        '<geom type="hfield" hfield="terrain" material="material"/>'
        '<body name="a"><joint/><geom name="a" size=".1"/></body>'
        '<body name="b" pos="0 0 1"><joint/><geom name="b" size=".1"/></body>'
        "</worldbody></mujoco>"
    )
    if strippath:
        path.write_text(
            path.read_text()
            .replace("<compiler ", '<compiler strippath="true" ')
            .replace('file="shared.png"', 'file="old/path/shared.png"')
        )
    reference = mujoco.MjModel.from_xml_path(str(path))
    with pytest.raises(NotImplementedError, match="cube-face"):
        compile_native_template(
            (NativeAssetInstance("x", str(path)),), timestep=0.002, gravity=(0, 0, -9.81)
        )
    path.write_text(path.read_text().replace(f'<texture name="cube" type="cube" {cube}/>', ""))
    reference = mujoco.MjModel.from_xml_path(str(path))
    template = compile_native_template(
        (NativeAssetInstance("x", str(path)),), timestep=0.002, gravity=(0, 0, -9.81)
    )
    model = template.compilation.mj_model
    for name in (
        "hfield_data",
        "tex_data",
        "hfield_nrow",
        "hfield_ncol",
        "tex_height",
        "tex_width",
    ):
        np.testing.assert_array_equal(getattr(model, name), getattr(reference, name), err_msg=name)
    source = template.compilation.collision_source
    keys = {r.geom_name: r.key for r in source.geoms}
    plan = source.plan_exclusions(((keys["x::a"], keys["x::b"]),), source_identity=source.identity)
    for directory in (heights, images):
        for asset in directory.iterdir():
            asset.unlink()
        directory.rmdir()
    path.unlink()
    filtered = template.with_collision_exclusions(plan)
    for name in ("hfield_data", "tex_data"):
        np.testing.assert_array_equal(
            getattr(filtered.compilation.mj_model, name), getattr(reference, name)
        )


def test_native_strippath_mesh_matches_compiler(tmp_path):
    folder = tmp_path / "meshes"
    folder.mkdir()
    (folder / "tetra.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    path = tmp_path / "model.xml"
    path.write_text(
        '<mujoco><compiler strippath="true" meshdir="meshes"/>'
        '<asset><mesh name="m" file="old/path/tetra.obj"/></asset>'
        '<worldbody><body><joint/><geom type="mesh" mesh="m"/></body></worldbody></mujoco>'
    )
    reference = mujoco.MjModel.from_xml_path(str(path))
    template = compile_native_template(
        (NativeAssetInstance("x", str(path)),), timestep=0.002, gravity=(0, 0, -9.81)
    )
    for name in ("mesh_vert", "mesh_face", "body_mass", "body_inertia"):
        np.testing.assert_array_equal(
            getattr(template.compilation.mj_model, name), getattr(reference, name)
        )

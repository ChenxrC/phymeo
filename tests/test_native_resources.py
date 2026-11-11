"""Cold rule recompilation owns directory-qualified and cube resources."""

import struct
import zlib

import mujoco
import numpy as np
import pytest

from phymeo.builders.native_assembly import NativeAssetInstance, compile_native_template


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

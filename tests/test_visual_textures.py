"""Compiled texture snapshots retain bytes, roles and projection independently of files."""

import mujoco
import numpy as np
import pytest

from phymeo.builders.native_visuals import export_model


def test_compiled_textures_preserve_pixels_material_and_environment():
    model = mujoco.MjModel.from_xml_string("""<mujoco><asset>
      <texture name="flat" type="2d" builtin="checker" width="4" height="4"/>
      <texture name="cube" type="cube" builtin="checker" width="4" height="4"/>
      <texture name="sky" type="skybox" builtin="gradient" width="4" height="4"/>
      <material name="m" texture="flat" texrepeat="2 3" texuniform="true"/>
      </asset><worldbody><geom name="box" type="box" size="1 1 1" material="m"/></worldbody></mujoco>""")
    before = model.tex_data.copy()
    scene = export_model(model)
    assert scene.contract_version == 2
    assert [t.kind for t in scene.textures] == ["2d", "cube", "skybox"]
    assert [t.source_name for t in scene.textures] == ["flat", "cube", "sky"]
    for index, texture in enumerate(scene.textures):
        start = model.tex_adr[index]
        np.testing.assert_array_equal(
            texture.pixels.ravel(), before[start : start + texture.pixels.size]
        )
        with pytest.raises(ValueError):
            texture.pixels.setflags(write=True)
    material = dict(scene.instances[0].material)
    assert material["texture_repeat"] == (2, 3)
    assert material["texture_uniform"] is True
    assert material["texture_bindings"] == ((1, "texture_0"),)
    model.tex_data[:] = 0
    assert np.any(scene.textures[0].pixels)


def test_untextured_contract_unchanged():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom size="1"/></worldbody></mujoco>'
    )
    assert export_model(model).contract_version == 1

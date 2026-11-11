"""Solver-specific asset assembly, identity and once-only homogeneous binding."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from phymeo.builders.native_assembly import (
    NativeAssetInstance,
    NativeJointInput,
    NativeStaticShape,
    compile_native_template,
)

URDF = """<robot name="slider"><link name="base"/>
<link name="link"><inertial><mass value="2"/><inertia ixx=".1" iyy=".1" izz=".1" ixy="0" ixz="0" iyz="0"/></inertial>
<collision><geometry><box size=".2 .2 .2"/></geometry></collision></link>
<joint name="slide" type="prismatic"><parent link="base"/><child link="link"/>
<axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="3" velocity="2"/></joint></robot>"""


@pytest.fixture
def urdf(tmp_path):
    path = tmp_path / "slider.urdf"
    path.write_text(URDF)
    return str(path)


def template(path):
    row = NativeJointInput(
        "slide",
        position=0.125,
        velocity=0.25,
        target_position=0.375,
        target_velocity=0.125,
        applied_force=15,
        kp=2,
        kd=0.5,
        effort_limit=0,
    )
    asset = NativeAssetInstance(
        "arm", path, position=(1.0, 2.0, 3.0), joints=(row,), drive_mode="scalar_pd"
    )
    return compile_native_template((asset,), timestep=0.002, gravity=(0, -9.81, 0))


def test_urdf_pose_control_binding_and_zero_effort(urdf):
    import mujoco

    result = template(urdf)
    model = result.compilation.fresh_native_model()
    data = mujoco.MjData(model)
    result.initial_state.apply(result.compilation, data)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos[model.body("arm::link").id], [1.125, 2, 3], atol=1e-10)
    assert result.joint_bindings == (("arm::slide", 0, 0, 0, 0),)
    np.testing.assert_array_equal(data.qvel, [0.25])
    np.testing.assert_array_equal(data.ctrl, [0.8125])
    np.testing.assert_array_equal(data.qfrc_applied, [15])
    assert model.actuator_forcelimited[0] == 0
    np.testing.assert_array_equal(model.qpos0, [0])
    np.testing.assert_array_equal(model.dof_armature, [0.01])
    assert result.qualification.startswith("native-only")


def test_two_instances_one_model_environment_and_binding_without_source(urdf, monkeypatch):
    asset = NativeAssetInstance("a", urdf, drive_mode="scalar_pd")
    result = compile_native_template(
        (asset, replace(asset, name="b", position=(0, 1, 0))),
        timestep=0.001,
        gravity=(0, 0, -9.81),
        static_shapes=(NativeStaticShape("table", "box", (1, 1, 0.1)),),
    )
    assert result.compilation.mj_model.nv == 2
    assert len(result.shape_bindings) == 3

    def forbidden(*args, **kwargs):
        raise AssertionError("binding cannot reread assets or recompile")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr("phymeo.builders.native_assembly.compile_mujoco_spec", forbidden)
    for count in (1, 8, 64):
        bound = result.bind(count)
        assert bound.template is result
        assert bound.nworld == count
        assert result.compilation.fresh_native_model().nv == 2
    with pytest.raises(ValueError):
        result.bind(True)


def test_native_actuator_is_preserved_and_never_duplicated(tmp_path):
    path = tmp_path / "driven.xml"
    path.write_text("""<mujoco><worldbody><body name="b"><joint name="j" type="slide"/>
    <geom type="sphere" size=".1"/></body></worldbody>
    <actuator><motor name="motor" joint="j" gear="2"/></actuator></mujoco>""")
    asset = NativeAssetInstance("drive", str(path))
    result = compile_native_template((asset,), timestep=0.002, gravity=(0, 0, 0))
    model = result.compilation.mj_model
    assert model.nu == 1 and model.actuator("drive::motor").id == 0
    assert model.actuator_gear[0, 0] == 2
    with pytest.raises(ValueError, match="duplicate source actuators"):
        compile_native_template(
            (replace(asset, drive_mode="scalar_pd"),), timestep=0.002, gravity=(0, 0, 0)
        )


def test_unknown_joint_and_mimic_are_not_dropped(urdf, tmp_path):
    asset = NativeAssetInstance(
        "a", urdf, joints=(NativeJointInput("missing"),), drive_mode="scalar_pd"
    )
    with pytest.raises(ValueError, match="unknown joint"):
        compile_native_template((asset,), timestep=0.002, gravity=(0, 0, 0))
    path = tmp_path / "mimic.urdf"
    path.write_text(URDF.replace("<axis", '<mimic joint="slide"/><axis'))
    with pytest.raises(NotImplementedError, match="mimic"):
        compile_native_template(
            (replace(asset, path=str(path), joints=()),), timestep=0.002, gravity=(0, 0, 0)
        )
    with pytest.raises(ValueError, match="ignore"):
        NativeAssetInstance("a", urdf, joint_defaults=(("kp", 20.0),))
    with pytest.raises(ValueError, match="ignore nonzero friction"):
        NativeStaticShape("table", "box", (1, 1, 0.1), contact_dim=3)


def test_fanuc_original_meshes_and_contact_policy_survive_assembly():
    import mujoco
    from test_physolve_models import robot_path

    path = robot_path("fanuc")
    reference = mujoco.MjModel.from_xml_path(str(path))
    asset = NativeAssetInstance("fanuc", str(path), drive_mode="scalar_pd")
    result = compile_native_template(
        (asset,), timestep=reference.opt.timestep, gravity=tuple(reference.opt.gravity)
    )
    model = result.compilation.fresh_native_model()
    assert model.nmesh == reference.nmesh and model.ngeom == reference.ngeom
    assert model.nv == reference.nv == 6
    data, expected = mujoco.MjData(model), mujoco.MjData(reference)
    mujoco.mj_forward(model, data)
    mujoco.mj_forward(reference, expected)
    # Static fusion may change body IDs, not geometry or its collision behavior.
    np.testing.assert_allclose(data.geom_xpos, expected.geom_xpos, atol=1e-10, rtol=0)
    np.testing.assert_allclose(data.geom_xmat, expected.geom_xmat, atol=1e-10, rtol=0)
    assert {tuple(c.geom) for c in data.contact} == {tuple(c.geom) for c in expected.contact}
    assert len(result.compilation.collision_source.geoms) == model.ngeom
    assert not result.compilation.collision_source.issues


def test_include_mesh_refs_and_binary_survive_source_removal(tmp_path):
    mesh = tmp_path / "tetra.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
    include = tmp_path / "body.xml"
    include.write_text(
        '<mujoco><body name="body"><joint name="hinge"/><geom mesh="tetra" type="mesh"/></body></mujoco>'
    )
    source = tmp_path / "main.xml"
    source.write_text("""<mujoco><asset><mesh name="tetra" file="tetra.obj"/></asset>
    <worldbody><include file="body.xml"/></worldbody>
    <actuator><motor name="motor" joint="hinge"/></actuator></mujoco>""")
    a = NativeAssetInstance("a", str(source))
    result = compile_native_template(
        (a, replace(a, name="b", position=(3, 0, 0))), timestep=0.001, gravity=(0, 0, -9.81)
    )
    expected = result.compilation.mj_model
    for path in (mesh, include, source):
        path.unlink()
    restored = result.compilation.fresh_native_model()
    assert restored.nu == 2 and restored.nmesh == 2
    for name in ("mesh_vert", "mesh_face", "geom_dataid", "actuator_trnid", "body_pos"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(expected, name))
    assert restored.actuator("b::motor").trnid[0] == restored.joint("b::hinge").id


@pytest.mark.parametrize("option", ['cone="elliptic"', 'iterations="7"'])
def test_source_global_options_cannot_be_silently_discarded(tmp_path, option):
    path = tmp_path / "option.xml"
    path.write_text(
        f'<mujoco><option {option}/><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    with pytest.raises(NotImplementedError, match="source option"):
        compile_native_template(
            (NativeAssetInstance("a", str(path)),), timestep=0.001, gravity=(0, 0, 0)
        )

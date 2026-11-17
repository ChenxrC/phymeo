"""Solver-specific asset assembly, identity and once-only homogeneous binding."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from phymeo.builders.native_assembly import (
    NativeAssetInstance,
    NativeJointInput,
    NativeRigidShape,
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


JOINT_SCALAR_FIELDS = tuple(
    name for name in NativeJointInput.__dataclass_fields__ if name != "name"
)


@pytest.mark.parametrize("field", JOINT_SCALAR_FIELDS)
@pytest.mark.parametrize("value", [
    True, np.bool_(False), 1 + 2j, np.complex64(2), "2", [2.0],
    np.array(2.0), np.array([2.0]), np.array([2.0, 3.0]),
    np.nan, np.inf, -np.inf, 1e39, 10**1000,
])
def test_joint_input_rejects_nonreal_containers_and_out_of_range(field, value):
    with pytest.raises(ValueError, match="finite scalar"):
        NativeJointInput("slide", **{field: value})


@pytest.mark.parametrize("field", ["kp", "kd", "armature", "damping", "frictionloss"])
@pytest.mark.parametrize("value", [True, 1 + 0j, np.array(2.), np.array([2.]),
                                  [2.], "2", np.nan, np.inf, -1.])
def test_joint_defaults_reject_invalid_scalars(urdf, field, value):
    with pytest.raises(ValueError):
        NativeAssetInstance("arm", urdf, drive_mode="scalar_pd",
                            joint_defaults=((field, value),))


def test_joint_inputs_and_defaults_freeze_caller_values(urdf):
    from dataclasses import FrozenInstanceError

    source = np.array([.125, .25, 4.], dtype=np.float32)
    arguments = dict(position=source[0], velocity=source[1], applied_force=source[2])
    row = NativeJointInput("slide", **arguments)
    rows = [row]
    defaults = [["kp", np.float32(20)], ["kd", np.float64(2)]]
    asset = NativeAssetInstance("arm", urdf, drive_mode="scalar_pd",
                                joints=rows, joint_defaults=defaults)
    source[:] = np.nan
    arguments["position"] = 99.
    rows.clear()
    defaults[0][1] = np.nan
    defaults.clear()
    assert (row.position, row.velocity, row.applied_force) == (.125, .25, 4.)
    assert asset.joints == (row,)
    assert asset.joint_defaults == (("kp", 20.), ("kd", 2.))
    assert all(type(getattr(row, key)) is float for key in arguments)
    assert all(type(value) is float for _, value in asset.joint_defaults)
    with pytest.raises(FrozenInstanceError):
        row.position = 99.


def test_python_and_numpy_scalar_models_are_identical(urdf):
    import mujoco

    values = dict(position=.125, velocity=.25, target_position=.5, target_velocity=.75,
                  applied_force=2., kp=20., kd=2., armature=.125, damping=.25,
                  frictionloss=.125, effort_limit=3., lower_limit=-.5,
                  upper_limit=.75, velocity_limit=2.)
    for scalar in (float, int, np.float32, np.float64, np.int32, np.int64):
        # Integral inputs exercise the same contract with exactly representable values.
        selected = values if scalar in (float, np.float32, np.float64) else {
            key: int(value) for key, value in values.items()
        }
        if scalar in (int, np.int32, np.int64):
            selected.update(lower_limit=-1, upper_limit=1)
        row = NativeJointInput("slide", **{key: scalar(v) for key, v in selected.items()})
        defaults = (("kp", scalar(8)), ("kd", scalar(1)))
        actual = compile_native_template((NativeAssetInstance("arm", urdf,
            drive_mode="scalar_pd", joints=(row,), joint_defaults=defaults),),
            timestep=.002, gravity=(0, 0, 0))
        reference = compile_native_template((NativeAssetInstance("arm", urdf,
            drive_mode="scalar_pd", joints=(NativeJointInput("slide", **selected),),
            joint_defaults=(("kp", 8.), ("kd", 1.))),), timestep=.002, gravity=(0, 0, 0))
        a, b = actual.compilation.fresh_native_model(), reference.compilation.fresh_native_model()
        for name in dir(a):
            value = getattr(a, name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, getattr(b, name), err_msg=name)
        assert actual.scalar_joint_bindings == reference.scalar_joint_bindings
        assert actual.target_bindings == reference.target_bindings
        da, db = mujoco.MjData(a), mujoco.MjData(b)
        actual.initial_state.apply(actual.compilation, da)
        reference.initial_state.apply(reference.compilation, db)
        for field in ("qpos", "qvel", "ctrl", "qfrc_applied"):
            np.testing.assert_array_equal(getattr(da, field), getattr(db, field))


def test_source_joint_catalog_needs_no_mesh_or_native_model(urdf):
    from phymeo.builders.native_resources import read_urdf_joint_catalog
    snapshot = URDF.replace('<box size=".2 .2 .2"/>', '<mesh filename="missing.stl"/>')
    asset = NativeAssetInstance('arm', urdf, urdf_xml=snapshot)
    Path(urdf).unlink()
    assert read_urdf_joint_catalog(asset) == (('slide', 'prismatic', -1., 1.),)


def test_body_friction_targets_direct_source_shapes_without_changing_other_parameters(tmp_path):
    source = tmp_path / "material.xml"
    source.write_text('''<mujoco><worldbody><body name="base">
      <geom name="a" size=".1" friction=".4 .02 .003"/>
      <geom name="b" size=".1" pos=".3 0 0" friction=".6 .03 .004"/>
      <body name="child" pos="0 0 .5"><geom name="c" size=".1"/></body>
      </body><body name="empty"/></worldbody></mujoco>''')
    plain = NativeAssetInstance("arm", str(source))
    original = compile_native_template((plain,), timestep=.002, gravity=(0, 0, 0))
    rows = [["base", 1.5]]
    changed = replace(plain, body_friction=rows)
    rows[0][1] = .1
    template = compile_native_template((changed, replace(plain, name="neighbor")),
                                       timestep=.002, gravity=(0, 0, 0))
    before = original.compilation.fresh_native_model()
    after = template.compilation.fresh_native_model()
    for name in ("a", "b", "c"):
        expected = before.geom("arm::" + name).friction.copy()
        if name != "c":
            expected[0] = 1.5
        np.testing.assert_array_equal(after.geom("arm::" + name).friction, expected)
        np.testing.assert_array_equal(after.geom("neighbor::" + name).friction,
                                      before.geom("arm::" + name).friction)
        np.testing.assert_array_equal(after.geom("arm::" + name).solref,
                                      before.geom("arm::" + name).solref)
        assert after.geom("arm::" + name).condim == before.geom("arm::" + name).condim
    for name, message in (("missing", "unknown source body"), ("empty", "no shapes")):
        with pytest.raises(ValueError, match=message):
            compile_native_template((replace(plain, body_friction=((name, 1.),)),),
                                    timestep=.002, gravity=(0, 0, 0))


@pytest.mark.parametrize("rows", [(("base", -1.),), (("base", np.nan),),
    (("base", np.inf),), (("base", 1e40),), (("base", True),), (("base", "1"),),
    (("base", .4), ("base", .5)), (("arm::base", .4),), (("", .4),)])
def test_body_friction_rejects_invalid_declarations(urdf, rows):
    with pytest.raises(ValueError, match="friction"):
        NativeAssetInstance("arm", urdf, body_friction=rows)


def test_snapshot_is_authoritative_for_model_drive_visual_and_identity(urdf):
    snapshot = (
        URDF.replace('mass value="2"', 'mass value="4"')
        .replace('effort="3"', 'effort="7"')
        .replace(
            "<collision>", '<visual><geometry><sphere radius=".4"/></geometry></visual><collision>'
        )
    )
    asset = NativeAssetInstance("arm", urdf, drive_mode="scalar_pd", urdf_xml=snapshot)
    template = compile_native_template(
        (asset,), timestep=0.002, gravity=(0, 0, 0), capture_visuals=True
    )
    model = template.compilation.fresh_native_model()
    assert model.body("arm::link").mass == 4
    np.testing.assert_array_equal(model.actuator_forcerange[0], [-7, 7])
    assert template.compile_visual_scene().instances[0].size == (0.4,)
    assert template.asset_sources == (("arm", asset.path, sha256(snapshot.encode()).hexdigest()),)
    Path(urdf).unlink()
    # The same authoritative snapshot also compiles when no locator file exists.
    again = compile_native_template((asset,), timestep=0.002, gravity=(0, 0, 0))
    assert again.compilation.fresh_native_model().body("arm::link").mass == 4


def test_source_identity_preserves_renaming_slashes_and_multiple_instances(urdf):
    text = URDF.replace('name="slider"', 'name="source_model"')
    text = text.replace('"base"', '"/base"').replace('"link"', '"a/link"')
    instances = tuple(NativeAssetInstance(name, urdf, urdf_xml=text)
                      for name in ("renamed", "neighbor"))
    template = compile_native_template(instances, timestep=.002, gravity=(0, 0, 0))
    assert template.asset_identity_contract_version == 1
    model = template.compilation.fresh_native_model()
    assert len(template.asset_identities) == 2
    indices = []
    for identity, instance in zip(template.asset_identities, instances):
        assert (identity.instance, identity.format, identity.model_name) == (
            instance.name, "urdf", "source_model")
        assert tuple(row[0] for row in identity.bodies) == ("/base", "a/link")
        for source, compiled, index in identity.bodies:
            assert compiled == instance.name + "::" + source
            assert model.body(index).name == compiled
            indices.append(index)
    assert len(set(indices)) == 4
    Path(urdf).unlink()
    assert template.bind(64).template.asset_identities is template.asset_identities
    assert template.compilation.fresh_native_model().nbody == 5


@pytest.mark.parametrize("kind", ["prismatic", "revolute", "continuous"])
def test_compiled_source_frames_include_fixed_joints_and_com(urdf, kind):
    text = URDF.replace('type="prismatic"', f'type="{kind}"')
    text = text.replace('<inertial>', '<inertial><origin xyz=".03 .04 .05"/>')
    text = text.replace('<axis xyz=', '<origin xyz=".1 .2 .3" rpy="0 0 1.5707963267948966"/><axis xyz=')
    text = text.replace('</robot>', '<link name="tip"/><joint name="weld" type="fixed">'
                        '<parent link="link"/><child link="tip"/>'
                        '<origin xyz=".4 .5 .6"/></joint></robot>')
    assets = tuple(NativeAssetInstance(name, urdf, urdf_xml=text)
                   for name in ("first", "second"))
    template = compile_native_template(assets, timestep=.002, gravity=(0, 0, 0),
                                       capture_frames=True)
    frames = template.frame_metadata
    assert frames.contract_version == 1
    assert len(frames.joints) == 4
    for index, name in enumerate(("first", "second")):
        moving, fixed = frames.joints[index * 2:index * 2 + 2]
        assert moving.instance == name and moving.source_name == "slide"
        assert moving.kind == ("revolute" if kind == "continuous" else kind)
        assert moving.parent_body == name + "::base"
        assert moving.child_body == name + "::link"
        np.testing.assert_allclose(moving.parent_anchor, [.1, .2, .3, 0, 0, 2**-.5, 2**-.5],
                                   atol=1e-12, rtol=0)
        assert moving.child_anchor == (0, 0, 0, 0, 0, 0, 1)
        assert moving.axis_joint_local == (1, 0, 0)
        assert fixed.kind == "fixed" and fixed.axis_joint_local is None
        np.testing.assert_allclose(fixed.parent_anchor, [.4, .5, .6, 0, 0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(dict(frames.bodies)[name + "::link"], [.03, .04, .05], atol=1e-12)
    Path(urdf).unlink()
    assert template.bind(64).template.frame_metadata is frames


def test_source_frames_are_explicit_and_empty_template_is_not_missing():
    template = compile_native_template((), timestep=.002, gravity=(0, 0, 0))
    assert template.frame_metadata is None
    template = compile_native_template((), timestep=.002, gravity=(0, 0, 0), capture_frames=True)
    assert template.frame_metadata.bodies == template.frame_metadata.joints == ()


@pytest.mark.parametrize(
    "strip,filename", [(False, "meshes/shape.obj"), (True, "ignored/shape.obj")]
)
def test_snapshot_resource_roots_are_owned_for_collision_and_visual(tmp_path, strip, filename):
    root = tmp_path / "assets"
    meshdir = root if strip else root / "meshes"
    meshdir.mkdir(parents=True)
    mesh = meshdir / "shape.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
    shape = f'<mesh filename="{filename}" scale=".2 .3 .4"/>'
    snapshot = (
        URDF.replace('<box size=".2 .2 .2"/>', shape)
        .replace("<collision>", f"<visual><geometry>{shape}</geometry></visual><collision>")
        .replace(
            "</robot>", f'<mujoco><compiler strippath="{str(strip).lower()}"/></mujoco></robot>'
        )
    )
    asset = NativeAssetInstance(
        "arm", str(tmp_path / "missing/model.urdf"), urdf_xml=snapshot, resource_roots=(root,)
    )
    template = compile_native_template(
        (asset,), timestep=0.002, gravity=(0, 0, 0), capture_visuals=True
    )
    mesh.unlink()
    assert template.compilation.fresh_native_model().nmesh == 1
    assert len(template.compile_visual_scene().meshes) == 1
    assert template.compilation.fresh_native_model().ngeom == 1
    with pytest.raises(FileNotFoundError, match="resource roots"):
        compile_native_template((asset,), timestep=0.002, gravity=(0, 0, 0))


@pytest.mark.parametrize("value", ["", " ", 1, b"<robot/>"])
def test_snapshot_rejects_non_xml_text(urdf, value):
    with pytest.raises(ValueError, match="urdf_xml"):
        NativeAssetInstance("arm", urdf, urdf_xml=value)


def test_snapshot_rejects_mjcf_and_malformed_xml(urdf):
    from xml.etree import ElementTree as ET

    for text, error in (("<mujoco/>", ValueError), ("<robot>", ET.ParseError)):
        with pytest.raises(error):
            compile_native_template(
                (NativeAssetInstance("arm", urdf, urdf_xml=text),),
                timestep=0.002,
                gravity=(0, 0, 0),
            )


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("uri", ["missing.obj", "package://robot/missing.obj"])
def test_snapshot_missing_visual_is_isolated_from_physics(urdf, capture, uri):
    text = URDF.replace('<collision>',
        f'<visual><geometry><mesh filename="{uri}"/></geometry></visual><collision>')
    template = compile_native_template(
        (NativeAssetInstance("arm", urdf, urdf_xml=text),),
        timestep=.002, gravity=(0, 0, 0), capture_visuals=capture)
    assert template.compilation.fresh_native_model().ngeom == 1
    assert bool(template.visual_diagnostics) == capture
    if capture:
        with pytest.raises(NotImplementedError, match="visual export unavailable"):
            template.compile_visual_scene()
    physical = text.replace('<box size=".2 .2 .2"/>', f'<mesh filename="{uri}"/>')
    with pytest.raises((FileNotFoundError, NotImplementedError)):
        compile_native_template((NativeAssetInstance("arm", urdf, urdf_xml=physical),),
                                timestep=.002, gravity=(0, 0, 0))


def test_default_resource_policy_matches_file_snapshot_and_explicit_root(tmp_path):
    mesh = tmp_path / "meshes/shape.obj"
    mesh.parent.mkdir()
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
    path = tmp_path / "model.urdf"
    text = URDF.replace('<box size=".2 .2 .2"/>', '<mesh filename="meshes/shape.obj"/>')
    path.write_text(text)
    models = []
    for options in ({}, {"urdf_xml": text}, {"resource_roots": (tmp_path,)}):
        template = compile_native_template((NativeAssetInstance("arm", str(path), **options),),
                                           timestep=.002, gravity=(0, 0, 0))
        models.append(template.compilation.fresh_native_model())
    for name in dir(models[0]):
        if isinstance(getattr(models[0], name), np.ndarray):
            for other in models[1:]:
                np.testing.assert_array_equal(getattr(models[0], name), getattr(other, name),
                                              err_msg=name)


@pytest.mark.parametrize("size", [(0., 0., .1), (2., 3., .1)])
def test_static_plane_compiles_infinite_contact_with_public_identity(size):
    import mujoco

    plane = NativeStaticShape("floor", "plane", size, position=(0., 0., .3),
                              friction=(.6, .02, .003))
    ball = NativeRigidShape("ball", "sphere", (.1,), position=(4., 0., .39), mass=1.)
    template = compile_native_template((), timestep=.002, gravity=(0., 0., -9.81),
                                       static_shapes=(plane,), rigid_shapes=(ball,),
                                       capture_visuals=True)
    model = template.compilation.fresh_native_model()
    floor_id = model.geom("floor").id
    assert model.geom_type[floor_id] == mujoco.mjtGeom.mjGEOM_PLANE
    assert model.geom_bodyid[floor_id] == 0
    np.testing.assert_array_equal(model.geom_size[floor_id], size)
    np.testing.assert_array_equal(model.geom_friction[floor_id], plane.friction)
    data = mujoco.MjData(model)
    template.initial_state.apply(template.compilation, data)
    mujoco.mj_forward(model, data)
    # Contact beyond the finite visual rectangle still uses an infinite plane.
    assert data.ncon == 1
    np.testing.assert_allclose(data.contact[0].dist, -.01, atol=1e-12, rtol=0)
    assert set(data.contact[0].geom) == {floor_id, model.geom("ball").id}
    assert any(row[0] == "floor" for row in template.shape_bindings)


@pytest.mark.parametrize("size", [(-1., 1., .1), (0., 1., .1), (1., 1., 0.),
                                  (1., 1., float("nan"))])
def test_static_plane_rejects_invalid_display(size):
    with pytest.raises(ValueError):
        NativeStaticShape("floor", "plane", size)


def test_plane_cannot_be_dynamic():
    with pytest.raises(ValueError, match="static world ownership"):
        NativeRigidShape("floor", "plane", (0., 0., .1), mass=1.)


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
    with pytest.raises(ValueError, match="mimic"):
        compile_native_template(
            (replace(asset, path=str(path), joints=()),), timestep=0.002, gravity=(0, 0, 0)
        )
    with pytest.raises(ValueError, match="ignore"):
        NativeAssetInstance("a", urdf, joint_defaults=(("kp", 20.0),))


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


def test_integrator_selection_changes_only_compiled_option(urdf):
    import mujoco

    asset = NativeAssetInstance("a", urdf, drive_mode="scalar_pd")
    base = dict(timestep=0.002, gravity=(0, 0, -9.81))
    default = compile_native_template((asset,), **base)
    explicit = compile_native_template((asset,), **base, integrator="Euler")
    implicit = compile_native_template((asset,), **base, integrator="implicitfast")
    assert default.compilation.source.model_sha256 == explicit.compilation.source.model_sha256
    assert default.compilation.source.model_sha256 != implicit.compilation.source.model_sha256
    model = implicit.compilation.fresh_native_model()
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for name in dir(default.compilation.mj_model):
        if not name.startswith("_"):
            value = getattr(default.compilation.mj_model, name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, getattr(model, name), err_msg=name)
    for field in ("qpos", "qvel", "ctrl", "qfrc_applied"):
        first, second = mujoco.MjData(default.compilation.mj_model), mujoco.MjData(model)
        default.initial_state.apply(default.compilation, first)
        implicit.initial_state.apply(implicit.compilation, second)
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


@pytest.mark.parametrize("integrator", [None, True, 3, "rk4", "implicit", "unknown"])
def test_unknown_integrator_rejected_before_source_read(urdf, integrator, monkeypatch):
    import mujoco

    def forbidden(*args, **kwargs):
        raise AssertionError("must reject integrator before reading asset")
    monkeypatch.setattr(mujoco.MjSpec, "from_file", forbidden)
    with pytest.raises(ValueError, match="integrator"):
        compile_native_template((NativeAssetInstance("a", urdf),), timestep=.002,
                                gravity=(0, 0, 0), integrator=integrator)


def test_nondefault_source_integrator_requires_matching_request(tmp_path):
    import mujoco

    path = tmp_path / "implicit.xml"
    path.write_text('<mujoco><option integrator="implicitfast"/><worldbody>'
                    '<body><joint/><geom size=".1"/></body></worldbody></mujoco>')
    asset = NativeAssetInstance("a", str(path))
    with pytest.raises(NotImplementedError, match="source option integrator"):
        compile_native_template((asset,), timestep=.002, gravity=(0, 0, 0))
    result = compile_native_template((asset,), timestep=.002, gravity=(0, 0, 0),
                                     integrator="implicitfast")
    assert result.compilation.fresh_native_model().opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def test_rigid_only_mass_inertia_and_native_coordinate_bindings():
    import mujoco

    result = compile_native_template((), timestep=.002, gravity=(0,0,-9.81), rigid_shapes=(
        NativeRigidShape('ball','sphere',(.2,),position=(1,2,3),mass=2,density=900),
        NativeRigidShape('box','box',(.1,.2,.3),position=(-1,0,2),density=500),
    ), static_shapes=(NativeStaticShape('floor','box',(3,3,.1)),))
    model = result.compilation.fresh_native_model()
    assert (model.nq,model.nv,model.njnt) == (14,12,2)
    assert result.joint_dimensions == ((7,6),(7,6))
    assert result.scalar_joint_bindings == ()
    np.testing.assert_allclose(model.body_mass[1:], [2,24], rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.body_inertia[1], np.full(3,.032), rtol=0, atol=1e-12)
    data = mujoco.MjData(model)
    result.initial_state.apply(result.compilation,data)
    np.testing.assert_allclose(data.qpos[[0,1,2,7,8,9]], [1,2,3,-1,0,2], atol=1e-12)
    assert result.compilation.collision_source is not None


def test_floating_asset_preserves_internal_scalar_addresses(urdf):
    import mujoco

    path = Path(urdf)
    path.write_text(URDF.replace('<link name="base"/>', '<link name="base"><inertial><mass value="1"/><inertia ixx=".1" iyy=".1" izz=".1" ixy="0" ixz="0" iyz="0"/></inertial></link>'))
    base = NativeAssetInstance('arm',urdf,position=(1,2,3),drive_mode='scalar_pd',
                               joints=(NativeJointInput('slide',position=.125,velocity=.25),))
    fixed = compile_native_template((base,),timestep=.002,gravity=(0,0,0))
    floating = compile_native_template((replace(base,floating=True),),timestep=.002,gravity=(0,0,0))
    assert fixed.joint_dimensions == ((1,1),)
    assert floating.joint_dimensions == ((7,6),(1,1))
    assert floating.scalar_joint_bindings[0][2:4] == (7,6)
    model = floating.compilation.fresh_native_model()
    data = mujoco.MjData(model)
    floating.initial_state.apply(floating.compilation,data)
    mujoco.mj_forward(model,data)
    np.testing.assert_allclose(data.xpos[model.body('arm::link').id], [1.125,2,3], atol=1e-12)
    np.testing.assert_array_equal(data.qvel, [0,0,0,0,0,0,.25])
    assert floating.target_bindings == fixed.target_bindings


def test_empty_native_template_and_invalid_rigid_values():
    result = compile_native_template((),timestep=.002,gravity=(0,0,-9.81))
    assert result.scalar_joint_bindings == result.body_bindings == ()
    for values in (dict(mass=0),dict(mass=-1),dict(density=0),dict(density=None),dict(mass=np.inf)):
        with pytest.raises(ValueError):
            NativeRigidShape('body','sphere',(.1,),**values)
    with pytest.raises(TypeError,match='static_shapes'):
        compile_native_template((),timestep=.002,gravity=(0,0,0),
                                static_shapes=(NativeRigidShape('body','sphere',(.1,)),))


def test_two_floating_assets_and_rigid_keep_rotated_identity(urdf):
    import mujoco

    Path(urdf).write_text(URDF.replace('<link name="base"/>',
        '<link name="base"><inertial><mass value="1"/><inertia ixx=".1" iyy=".1" '
        'izz=".1" ixy="0" ixz="0" iyz="0"/></inertial></link>'))
    rotation = (0., 0., np.sqrt(.5), np.sqrt(.5))
    first = NativeAssetInstance('first', urdf, drive_mode='scalar_pd', floating=True,
        joints=(NativeJointInput('slide', position=.125, velocity=.25),))
    second = replace(first, name='second', position=(2., 3., 4.), orientation_xyzw=rotation)
    result = compile_native_template((first, second), timestep=.002, gravity=(0, 0, 0),
        rigid_shapes=(NativeRigidShape('ball', 'sphere', (.1,), mass=1.),))
    assert [row[2:4] for row in result.scalar_joint_bindings] == [(7, 6), (15, 13)]
    assert result.joint_dimensions == ((7, 6), (1, 1), (7, 6), (1, 1), (7, 6))
    model = result.compilation.fresh_native_model()
    data = mujoco.MjData(model)
    result.initial_state.apply(result.compilation, data)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos[model.body('first::link').id], [.125, 0., 0.], atol=1e-12)
    np.testing.assert_allclose(data.xpos[model.body('second::link').id], [2., 3.125, 4.], atol=1e-12)
    np.testing.assert_array_equal(data.qvel[[6, 13]], .25)
    for candidate, message in ((replace(result, coordinate_contract_version=2), 'contract'),
                               (replace(result, joint_dimensions=()), 'contract')):
        with pytest.raises(ValueError, match=message):
            _ = candidate.scalar_joint_bindings
    with pytest.raises(NotImplementedError, match='ball or multi-DOF'):
        _ = replace(result, joint_dimensions=((4, 3),) + result.joint_dimensions[1:]).scalar_joint_bindings


MJCF_SCALAR = """<mujoco model="native_slider"><compiler angle="radian"/>
<worldbody><body name="link" pos=".2 .1 .3">
<joint name="slide" type="slide" axis="1 0 0" range="-1 1" damping=".4" armature=".02" frictionloss=".03"/>
<inertial pos=".03 -.02 .04" mass="2" diaginertia=".1 .2 .25"/>
<geom name="a" type="sphere" size=".1"/><geom name="b" type="box" size=".1 .2 .1" pos="0 .3 0"/>
</body></worldbody></mujoco>"""


def test_mjcf_scalar_initial_control_frames_and_source_parameters(tmp_path):
    import mujoco as mj

    from phymeo.builders.native_resources import read_native_joint_catalog
    source = tmp_path / "slider.xml"
    source.write_text(MJCF_SCALAR)
    asset = NativeAssetInstance("arm", str(source), drive_mode="scalar_pd",
        joints=(NativeJointInput("slide", position=.125, velocity=.2,
                                target_position=.3, target_velocity=.4, applied_force=4., kp=10., kd=2.),))
    assert read_native_joint_catalog(asset) == (("slide", "prismatic", -1., 1.),)
    result = compile_native_template((asset,), timestep=.002, gravity=(0, 0, 0),
                                     capture_frames=True, capture_visuals=True)
    model = result.compilation.fresh_native_model()
    data = mj.MjData(model)
    result.initial_state.apply(result.compilation, data)
    np.testing.assert_array_equal(data.qpos, [.125])
    np.testing.assert_array_equal(data.qvel, [.2])
    np.testing.assert_array_equal(data.qfrc_applied, [4.])
    np.testing.assert_allclose(data.ctrl, [3.8], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(model.dof_armature, [.02])
    np.testing.assert_array_equal(model.dof_damping, [.4])
    np.testing.assert_array_equal(model.dof_frictionloss, [.03])
    assert result.asset_identities[0].format == "mjcf"
    assert result.asset_identities[0].bodies == (("link", "arm::link", 1),)
    assert tuple(n for n, _ in result.shape_bindings) == ("arm::a", "arm::b")
    frame = result.frame_metadata.joints[0]
    assert (frame.parent_body, frame.child_body, frame.kind) == ("world", "arm::link", "prismatic")
    np.testing.assert_array_equal(frame.parent_anchor[:3], [.2, .1, .3])
    np.testing.assert_array_equal(result.frame_metadata.bodies[0][1], [.03, -.02, .04])
    for count in (1, 8, 64):
        assert result.bind(count).nworld == count
    assert result.compile_visual_scene().instances


@pytest.mark.parametrize("extra", ["degree", "actuator", "keyframe", "ball", "nonlinear_damping", "springdamper"])
def test_mjcf_scalar_rejects_unqualified_source_semantics(tmp_path, extra):
    text = MJCF_SCALAR
    if extra == "degree":
        text = text.replace('angle="radian"', 'angle="degree"')
    elif extra == "actuator":
        text = text.replace("</mujoco>", '<actuator><motor joint="slide"/></actuator></mujoco>')
    elif extra == "keyframe":
        text = text.replace("</mujoco>", '<keyframe><key qpos=".2"/></keyframe></mujoco>')
    elif extra == "nonlinear_damping":
        text = text.replace('damping=".4"', 'damping=".4 .2 .1"')
    elif extra == "springdamper":
        text = text.replace('damping=".4"', 'damping=".4" springdamper=".02 .8"')
    else:
        text = text.replace('type="slide"', 'type="ball"')
    source = tmp_path / "unsupported.xml"
    source.write_text(text)
    reasons = {"degree": "angle=radian", "actuator": "source actuators", "keyframe": "keyframes",
               "ball": "scalar hinge/slide", "nonlinear_damping": "nonlinear damping", "springdamper": "springdamper"}
    with pytest.raises((ValueError, NotImplementedError), match=reasons[extra]):
        compile_native_template((NativeAssetInstance("arm", str(source), drive_mode="scalar_pd"),),
                                timestep=.002, gravity=(0, 0, 0), capture_frames=True)


@pytest.mark.parametrize("kind", ["hinge", "slide"])
def test_mjcf_nonzero_reference_anchor_reconstructs_native_pose(tmp_path, kind):
    import mujoco as mj

    from phymeo.dynamics.rotation import quaternion_matrix
    text = MJCF_SCALAR.replace('type="slide"', 'type="' + kind + '" ref=".3"').replace('axis="1 0 0"', 'axis="1 0 0" pos=".02 .04 -.01"')
    source = tmp_path / "reference.xml"
    source.write_text(text)
    result = compile_native_template((NativeAssetInstance("arm", str(source),
        orientation_xyzw=(0., 0., .6, .8), position=(1., 2., 3.), drive_mode="scalar_pd",
        joints=(NativeJointInput("slide", position=.6),)),), timestep=.002, gravity=(0, 0, 0), capture_frames=True)
    model = result.compilation.fresh_native_model()
    data = mj.MjData(model)
    result.initial_state.apply(result.compilation, data)
    mj.mj_kinematics(model, data)
    frame = result.frame_metadata.joints[0]
    parent = quaternion_matrix(np.array(frame.parent_anchor[3:])[None])[0]
    offset = np.array(frame.parent_anchor[:3])
    if kind == "hinge":
        q = np.empty(4)
        mj.mju_axisAngle2Quat(q, np.array(frame.axis_joint_local), .6)
        rotation = parent @ quaternion_matrix(q[[1, 2, 3, 0]][None])[0]
    else:
        offset += parent @ (np.array(frame.axis_joint_local) * .6)
        rotation = parent
    position = offset - rotation @ np.array(frame.child_anchor[:3])
    np.testing.assert_allclose(position, data.xpos[1], atol=1e-12, rtol=0)
    np.testing.assert_allclose(rotation, data.xmat[1].reshape(3, 3), atol=1e-12, rtol=0)


def test_primitive_contact_dimension_preserves_inactive_material():
    for dimension in (1, 3, 4, 6):
        for shape_type, keyword in ((NativeStaticShape, "static_shapes"), (NativeRigidShape, "rigid_shapes")):
            shape = shape_type("cube", "box", (.1, .1, .1), contact_dim=dimension)
            template = compile_native_template((), timestep=.01, gravity=(0, -9.81, 0), **{keyword: (shape,)})
            model = template.compilation.fresh_native_model()
            assert model.geom("cube").condim == dimension
            np.testing.assert_array_equal(model.geom("cube").friction, (1., .005, .0001))
    for invalid in (True, 3., "3", None, 0, 2, 5, 7):
        with pytest.raises(ValueError, match="contact_dim"):
            NativeStaticShape("cube", "box", (.1, .1, .1), contact_dim=invalid)
    default = NativeRigidShape("cube", "box", (.1, .1, .1))
    a = compile_native_template((), timestep=.01, gravity=(0, -9.81, 0), rigid_shapes=(default,)).compilation.fresh_native_model()
    b = compile_native_template((), timestep=.01, gravity=(0, -9.81, 0), rigid_shapes=(replace(default, contact_dim=6),)).compilation.fresh_native_model()
    for name in dir(a):
        if isinstance(value := getattr(a, name), np.ndarray):
            np.testing.assert_array_equal(value, getattr(b, name), err_msg=name)

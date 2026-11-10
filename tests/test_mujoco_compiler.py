"""Optional CPU compiler admission and independent MuJoCo kinematic/dynamic oracles.

AUTHORING_ONLY: these cases have not been imported, collected or executed by the author.
Missing MuJoCo skips only native oracle cases; dependency failure remains testable.
"""

import importlib
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from phymeo.builders import mujoco_compiler as compiler
from phymeo.builders.floating_urdf import FloatingRobotAsset
from phymeo.dynamics.floating_robot import (
    FloatingRobotState,
    floating_kinematics,
    floating_point_jacobians,
)
from phymeo.dynamics.robot_dynamics import mass_matrix
from phymeo.dynamics.robot_kinematics import forward_kinematics, point_jacobians
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.solvers.floating_tree import floating_mass_matrix

BODY = """<body name="base" pos=".3 -.2 .5" quat=".9238795325 0 0 .3826834324">
  {free}
  <inertial mass="2" pos=".07 -.04 .02" quat=".9238795325 .3826834324 0 0"
            diaginertia=".2 .3 .4"/>
  <geom name="base-sphere" type="sphere" size=".12" pos=".04 0 0"/>
  <body name="arm" pos=".2 .1 .15" quat=".9238795325 0 .3826834324 0">
    <joint name="turn" type="hinge" axis="0 1 0" pos=".03 -.02 .04"
           ref=".17" range="-.7 .9" {actuator}/>
    <inertial mass="1.3" pos=".06 .02 -.03" quat=".9238795325 0 0 .3826834324"
              diaginertia=".06 .08 .1"/>
    <geom name="arm-capsule" type="capsule" size=".04 .15" pos=".1 0 .1"
          quat=".9238795325 0 .3826834324 0"/>
    <body name="slider" pos=".1 .03 .2">
      <joint name="slide" type="slide" axis="1 0 0" pos="-.02 .01 0"
             ref=".08" range="-.2 .4" {actuator}/>
      <geom name="slider-box" type="box" size=".07 .09 .11" density="730"
            pos=".01 .02 .03" quat=".9238795325 .3826834324 0 0"/>
    </body>
  </body>
</body>"""


def xml_tree(floating=False, actuators=False, extra=""):
    body = BODY.format(
        free='<freejoint name="root-free"/>' if floating else "",
        actuator='armature=".13" damping=".21"' if actuators else "",
    )
    return f"""<mujoco model="compiler-oracle">
      <compiler angle="radian" fusestatic="false"/>
      <option gravity=".2 -.3 -9.5"/>
      <default><joint limited="true"/><geom contype="2" conaffinity="3"/></default>
      <worldbody>{body}</worldbody>{extra}</mujoco>"""


def native():
    return pytest.importorskip("mujoco", reason="optional CPU MuJoCo oracle is not installed")


def test_missing_optional_dependency_has_explicit_error(monkeypatch, tmp_path):
    original = importlib.import_module

    def unavailable(name, *args, **kwargs):
        if name == "mujoco":
            raise ModuleNotFoundError("test missing optional mujoco")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    with pytest.raises(ImportError, match="optional MuJoCo"):
        compiler.compile_mujoco(tmp_path / "unused.xml")


@pytest.mark.parametrize("floating,actuators", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("offset", [0.0, 0.23, -0.31])
def test_independent_fk_full_mass_and_point_jacobians(floating, actuators, offset):
    mj = native()
    xml = xml_tree(floating, actuators)
    record = compiler.compile_mujoco_xml(xml)
    reference = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(reference)
    q = record.initial_q.copy()
    q[-2:] += [offset, -0.4 * offset]
    if floating:
        q[:3] += [0.1, -0.2, 0.3]
        q[3:7] = [0.2, -0.1, 0.3, np.sqrt(0.86)]
    velocity = np.linspace(-0.3, 0.4, reference.nv)
    data.qpos[:] = record.q_to_qpos(q)
    data.qvel[:] = record.v_to_qvel(q, velocity)
    mj.mj_forward(reference, data)
    np.testing.assert_allclose(record.qpos_to_q(data.qpos), q, atol=3e-15)
    np.testing.assert_allclose(record.qvel_to_v(q, data.qvel), velocity, atol=3e-15)
    transform = record.velocity_map(q)
    dense = np.empty((reference.nv, reference.nv))
    mj.mj_fullM(reference, dense, data.qM)
    if floating:
        state = FloatingRobotState(q, velocity)
        geometry = floating_kinematics(record.asset.model, state)
        physical_mass = floating_mass_matrix(record.asset.model, state)
    else:
        geometry = forward_kinematics(record.asset.model, q)
        physical_mass = mass_matrix(record.asset.model, q)
    np.testing.assert_allclose(
        physical_mass, transform.T @ dense @ transform, rtol=2e-12, atol=2e-12
    )
    for body in record.mapping.bodies:
        if body.link_index is None or (not floating and body.link_index == -1):
            continue
        index = body.link_index + 1 if floating else body.link_index
        point = np.array([0.025, -0.018, 0.04])
        origin = np.array(body.authored_origin_in_link)
        local = origin + quaternion_matrix(np.array([body.principal_to_authored_xyzw]))[0].T @ point
        expected_point = data.xpos[body.mj_id] + data.xmat[body.mj_id].reshape(3, 3) @ point
        actual_point = geometry.position[index] + geometry.rotation[index] @ local
        np.testing.assert_allclose(actual_point, expected_point, atol=2e-12)
        jp, jr = np.zeros((3, reference.nv)), np.zeros((3, reference.nv))
        mj.mj_jac(reference, data, jp, jr, expected_point, body.mj_id)
        if floating:
            actual_jp, actual_jr = floating_point_jacobians(
                record.asset.model, state, [index], local[None]
            )
        else:
            actual_jp, actual_jr = point_jacobians(record.asset.model, q, [index], local[None])
        np.testing.assert_allclose(actual_jp[0], jp @ transform, atol=2e-12)
        np.testing.assert_allclose(actual_jr[0], jr @ transform, atol=2e-12)
    turn, slide = record.mapping.joints
    assert record.asset.joint_names == ("turn", "slide")
    assert turn.qpos_address == (7 if floating else 0)
    assert slide.dof_address == (7 if floating else 1)
    np.testing.assert_allclose(record.asset.lower, [-0.87, -0.28], atol=1e-14)
    np.testing.assert_allclose(record.asset.upper, [0.73, 0.32], atol=1e-14)
    if actuators:
        np.testing.assert_array_equal(record.asset.model.joint_armature, [0.13, 0.13])
        np.testing.assert_array_equal(record.asset.model.joint_damping, [0.21, 0.21])


def test_defaults_include_inferred_inertia_and_owned_source_snapshot(tmp_path):
    mj = native()
    included = tmp_path / "part.xml"
    included.write_text("""<mujoco><worldbody><body name="payload"><freejoint/>
      <geom name="solid" type="box" size=".1 .2 .3" density="500"/>
      </body></worldbody></mujoco>""")
    main = tmp_path / "model.xml"
    main.write_text("""<mujoco><compiler angle="radian"/><include file="part.xml"/></mujoco>""")
    record = compiler.compile_mujoco(main)
    assert isinstance(record.asset, FloatingRobotAsset)
    assert record.asset.model.children is None
    assert record.asset.model.root.inverse_mass[0] == pytest.approx(1 / 24)
    rotation = quaternion_matrix(record.asset.model.root_inertia_frame_xyzw[None])[0]
    np.testing.assert_allclose(
        rotation @ np.diag(record.asset.model.root.inertia_diagonal[0]) @ rotation.T,
        np.diag([1.04, 0.8, 0.4]),
        atol=1e-12,
    )
    included.unlink()
    main.unlink()
    restored = record.fresh_native_model()
    assert restored.nbody == record.mj_model.nbody
    np.testing.assert_array_equal(restored.body_mass, record.mj_model.body_mass)
    assert restored is not record.mj_model
    restored.body_mass[1] = 123
    assert record.mj_model.body_mass[1] != 123
    assert record.source.primary_xml.startswith(b"<mujoco>")
    assert record.source.archive_sha256
    assert record.source.mujoco_version == mj.__version__
    with pytest.raises(FrozenInstanceError):
        record.source.archive_sha256 = "changed"
    with pytest.raises(ValueError):
        record.initial_q.flags.writeable = True


def test_default_joint_actuators_and_visual_physical_geometries_are_preserved():
    native()
    xml = xml_tree(True).replace(
        '<joint limited="true"/>', '<joint limited="true" armature=".3" damping=".4"/>'
    )
    xml = xml.replace(
        '<geom name="base-sphere"',
        '<geom name="visual-only" type="sphere" size=".2" contype="0" conaffinity="0"/><geom name="base-sphere"',
    )
    record = compiler.compile_mujoco_xml(xml)
    np.testing.assert_array_equal(record.asset.model.joint_armature, [0.3, 0.3])
    np.testing.assert_array_equal(record.asset.model.joint_damping, [0.4, 0.4])
    assert {row.name for row in record.asset.collisions} == {
        "base-sphere",
        "arm-capsule",
        "slider-box",
    }
    visual = next(row for row in record.mapping.geoms if row.name == "visual-only")
    assert visual.shape_key is None and visual.visual_only
    assert all(row.shape.group == 2 and row.shape.mask == 3 for row in record.asset.collisions)


def test_world_environment_and_contact_law_are_explicit_metadata():
    native()
    xml = xml_tree(True).replace(
        "<worldbody>", '<worldbody><geom name="floor" type="plane" size="1 1 .1"/>'
    )
    record = compiler.compile_mujoco_xml(xml)
    floor = next(row for row in record.mapping.geoms if row.name == "floor")
    assert floor.world_static and floor.shape_key is None
    assert record.unlowered_world_geom_ids == (floor.mj_id,)
    assert (
        record.asset_scope
        == "articulation-only; environment and contact law require explicit scene assembly"
    )


@pytest.mark.parametrize(
    "change,issue",
    [
        (
            lambda x: x.replace('type="hinge"', 'type="ball" limited="false"').replace(
                'ref=".17" range="-.7 .9"', ""
            ),
            "joint-type",
        ),
        (lambda x: x.replace('name="turn"', 'name="turn" frictionloss=".2"'), "joint-frictionloss"),
        (lambda x: x.replace('name="turn"', 'name="turn" stiffness="2"'), "joint-stiffness"),
        (
            lambda x: x.replace(
                "</mujoco>", '<equality><joint joint1="turn" joint2="slide"/></equality></mujoco>'
            ),
            "equality",
        ),
        (
            lambda x: x.replace(
                "</mujoco>", '<actuator><motor joint="turn" gear="2"/></actuator></mujoco>'
            ),
            "actuator-transmission",
        ),
    ],
)
def test_valid_native_unsupported_semantics_do_not_silently_lower(change, issue):
    native()
    xml = change(xml_tree(True))
    record = compiler.compile_mujoco_xml(xml, require_asset=False)
    assert record.asset is None
    assert issue in {item.code for item in record.lowering_issues}
    assert record.mj_model.nv > 0
    with pytest.raises(compiler.MuJoCoLoweringError) as error:
        compiler.compile_mujoco_xml(xml)
    assert issue in {item.code for item in error.value.issues}


def test_fixed_joint_actuators_require_explicit_native_only_boundary():
    native()
    with pytest.raises(compiler.MuJoCoLoweringError, match="fixed-actuators"):
        compiler.compile_mujoco_xml(xml_tree(False, True))
    record = compiler.compile_mujoco_xml(xml_tree(False, True), require_asset=False)
    assert record.asset is None
    np.testing.assert_array_equal(record.mj_model.dof_armature, [0.13, 0.13])
    np.testing.assert_array_equal(record.mj_model.dof_damping, [0.21, 0.21])


def test_unit_motor_effort_limit_preserved_and_native_data_does_not_alias():
    native()
    xml = xml_tree(
        True,
        extra="""<actuator><motor name="motor" joint="turn" ctrlrange="-7 7"/>
      <motor name="slide-motor" joint="slide" ctrlrange="-3 3"/></actuator>""",
    )
    record = compiler.compile_mujoco_xml(xml)
    assert record.asset.effort_limit == (7.0, 3.0)
    before = record.native_initial_qpos.copy()
    record.mj_model.qpos0[:] = 999
    np.testing.assert_array_equal(record.native_initial_qpos, before)
    assert record.fresh_native_model().qpos0[0] != 999


def test_invalid_xml_is_compile_error_and_coordinate_shapes_are_checked():
    native()
    with pytest.raises(compiler.MuJoCoCompilationError):
        compiler.compile_mujoco_xml("<mujoco><not-a-real-tag/></mujoco>")
    record = compiler.compile_mujoco_xml(xml_tree(True))
    for method, args in (
        (record.q_to_qpos, (np.zeros(2),)),
        (record.qpos_to_q, (np.zeros(2),)),
        (record.v_to_qvel, (record.initial_q, np.full(8, np.nan))),
        (record.velocity_map, (np.zeros(9),)),
    ):
        with pytest.raises(ValueError):
            method(*args)


def test_full_inertia_tensor_is_decoded_without_diagonal_loss():
    mj = native()
    xml = xml_tree(True).replace(
        'quat=".9238795325 0 0 .3826834324"\n              diaginertia=".06 .08 .1"',
        'fullinertia=".06 .08 .1 .005 .003 -.002"',
    )
    record = compiler.compile_mujoco_xml(xml)
    reference = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(reference)
    data.qpos[:] = record.q_to_qpos(record.initial_q)
    mj.mj_forward(reference, data)
    dense = np.empty((reference.nv, reference.nv))
    mj.mj_fullM(reference, dense, data.qM)
    transform = record.velocity_map(record.initial_q)
    state = FloatingRobotState(record.initial_q, record.initial_v)
    np.testing.assert_allclose(
        floating_mass_matrix(record.asset.model, state), transform.T @ dense @ transform, atol=2e-12
    )


def test_asymmetric_source_or_collision_filter_is_not_silently_reinterpreted():
    native()
    xml = xml_tree(True).replace(
        'name="base-sphere"', 'name="base-sphere" contype="1" conaffinity="0"'
    )
    with pytest.raises(compiler.MuJoCoLoweringError, match="collision-filter"):
        compiler.compile_mujoco_xml(xml)


def test_native_tendon_and_multiple_articulations_have_explicit_capability_errors():
    native()
    tendon = '<tendon><fixed name="coupled"><joint joint="turn" coef="1"/></fixed></tendon>'
    with pytest.raises(compiler.MuJoCoLoweringError, match="tendon"):
        compiler.compile_mujoco_xml(xml_tree(True, extra=tendon))
    welded = xml_tree(True).replace(
        "</worldbody>",
        '<body name="other"><joint name="other-joint" limited="false"/>'
        '<geom size=".1"/></body></worldbody>',
    )
    with pytest.raises(compiler.MuJoCoLoweringError, match="multiple-articulations"):
        compiler.compile_mujoco_xml(welded)


def test_unfolded_fixed_child_has_an_explicit_capability_error():
    native()
    xml = xml_tree(True).replace(
        '<body name="slider"',
        '<body name="weld" pos=".2 0 0">'
        '<geom name="weld-shape" type="sphere" size=".02" mass=".2"/></body>'
        '<body name="slider"',
    )
    with pytest.raises(compiler.MuJoCoLoweringError, match="fixed-child"):
        compiler.compile_mujoco_xml(xml)


def test_virtual_include_bytes_and_unlimited_joint_metadata():
    native()
    part = (
        BODY.format(free='<freejoint name="root-free"/>', actuator="")
        .replace('range="-.7 .9"', 'limited="false"')
        .replace('range="-.2 .4"', 'limited="false"')
    )
    record = compiler.compile_mujoco_xml(
        '<mujoco><compiler angle="radian"/><include file="part.xml"/></mujoco>',
        includes={"part.xml": f"<mujoco><worldbody>{part}</worldbody></mujoco>".encode()},
    )
    assert record.asset.name == "MuJoCo Model"
    assert np.all(np.isneginf(record.asset.lower))
    assert np.all(np.isposinf(record.asset.upper))
    assert record.asset.effort_limit == (None, None)


def test_physical_unsupported_geometry_is_not_treated_as_visual():
    native()
    xml = xml_tree(True).replace(
        'name="arm-capsule" type="capsule"', 'name="arm-capsule" type="cylinder"'
    )
    with pytest.raises(compiler.MuJoCoLoweringError, match="geometry-type"):
        compiler.compile_mujoco_xml(xml)

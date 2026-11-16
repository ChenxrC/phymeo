"""Native URDF semantics: source dynamics and one affine physical constraint."""

from dataclasses import replace
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

from phymeo.builders.native_assembly import (
    NativeAssetInstance,
    NativeJointInput,
    compile_native_template,
)


def asset(tmp_path, *, ratio=-2, offset=0.1):
    path = tmp_path / "linkage.urdf"
    link = '<link name="{}"><inertial><mass value="1"/><inertia ixx=".1" iyy=".1" izz=".1" ixy="0" ixz="0" iyz="0"/></inertial></link>'
    joint = '<joint name="{}" type="prismatic"><parent link="base"/><child link="{}"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="10" velocity="3"/>{}</joint>'
    path.write_text(
        '<robot name="linkage"><link name="base"/>'
        + link.format("a")
        + link.format("b")
        + joint.format("driver", "a", '<dynamics damping=".25" friction=".05"/>')
        + joint.format(
            "follower", "b", f'<mimic joint="driver" multiplier="{ratio}" offset="{offset}"/>'
        )
        + "</robot>"
    )
    return NativeAssetInstance(
        "arm",
        str(path),
        drive_mode="scalar_pd",
        joints=(NativeJointInput("driver", position=0.2, velocity=0.3, kp=4, kd=1),),
    )


def compile_asset(value):
    return compile_native_template((value,), timestep=0.002, gravity=(0, 0, 0))


def test_native_affine_constraint_and_independent_commands(tmp_path):
    original = asset(tmp_path)
    result = compile_asset(original)
    model = result.compilation.fresh_native_model()
    assert (model.nq, model.nv, model.nu, model.neq) == (2, 2, 1, 1)
    assert model.eq_type[0] == mj.mjtEq.mjEQ_JOINT
    assert model.joint("arm::follower").id == model.eq_obj1id[0]
    assert model.joint("arm::driver").id == model.eq_obj2id[0]
    np.testing.assert_array_equal(model.eq_data[0, :5], [0.1, -2, 0, 0, 0])
    assert result.target_bindings == (("arm::driver", 0, 1.0, 0.0), ("arm::follower", 0, -2.0, 0.1))
    data = mj.MjData(model)
    result.initial_state.apply(result.compilation, data)
    np.testing.assert_allclose(data.qpos, [0.2, -0.3], atol=1e-12, rtol=0)
    np.testing.assert_allclose(data.qvel, [0.3, -0.6], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(model.dof_damping, [0.25, 0])
    np.testing.assert_array_equal(model.dof_frictionloss, [0.05, 1e-5])
    assert len(result.scalar_controls) == 1
    assert result.joint_bindings[1][-1] is None
    assert original.joints[0].damping is None
    override = replace(
        original,
        joint_defaults=(("damping", 0.6), ("frictionloss", 0.2)),
        joints=(replace(original.joints[0], damping=0.7),),
    )
    overridden = compile_asset(override).compilation.fresh_native_model()
    np.testing.assert_array_equal(overridden.dof_damping, [0.7, 0.6])
    np.testing.assert_array_equal(overridden.dof_frictionloss, [0.2, 0.2])


@pytest.mark.parametrize(
    "row",
    [
        NativeJointInput("follower", position=0.5),
        NativeJointInput("follower", velocity=0.5),
        NativeJointInput("follower", target_position=0.1),
    ],
)
def test_conflicting_follower_inputs_are_rejected(tmp_path, row):
    value = asset(tmp_path)
    with pytest.raises(ValueError, match="mimic"):
        compile_asset(replace(value, joints=(*value.joints, row)))


@pytest.mark.parametrize("field", ["target_position", "target_velocity"])
def test_initial_derived_target_overflow_is_rejected(tmp_path, field):
    value = asset(tmp_path, ratio=2, offset=0)
    row = replace(value.joints[0], kp=0, kd=0, **{field: 2e38})
    with pytest.raises(ValueError, match="initial derived target"):
        compile_asset(replace(value, joints=(row,)))


def test_passive_follower_accepts_external_force_without_a_motor(tmp_path):
    value = asset(tmp_path)
    compiled = compile_asset(
        replace(
            value, joints=(*value.joints, NativeJointInput("follower", kp=0, kd=0, applied_force=3))
        )
    )
    data = mj.MjData(compiled.compilation.mj_model)
    compiled.initial_state.apply(compiled.compilation, data)
    assert data.ctrl.size == 1
    np.testing.assert_array_equal(data.qfrc_applied, [0, 3])


def test_explicit_follower_servo_preserves_independent_command_contract(tmp_path):
    value = asset(tmp_path)
    compiled = compile_asset(replace(value, joints=(*value.joints,
        NativeJointInput('follower', kp=7, kd=2, effort_limit=3))))
    assert len(compiled.scalar_controls) == 1
    assert compiled.target_bindings == (('arm::driver',0,1.,0.),('arm::follower',0,-2.,.1))
    assert compiled.follower_drives == (('arm::follower',1,0,-2.,.1,7,2),)
    model = compiled.compilation.fresh_native_model()
    assert model.nu == 2 and model.neq == 1
    np.testing.assert_array_equal(model.actuator_biasprm[1,:3],[0,-7,-2])
    np.testing.assert_array_equal(model.actuator_forcerange[1],[-3,3])
    data = mj.MjData(model)
    compiled.initial_state.apply(compiled.compilation,data)
    np.testing.assert_allclose(data.ctrl,[.8,-2.1],rtol=0,atol=1e-12)


def test_prepared_follower_is_zero_gain_without_inheriting_defaults(tmp_path):
    value = asset(tmp_path)
    baseline = compile_asset(value)
    prepared = compile_asset(replace(value,prepare_follower_drives=True))
    assert baseline.follower_drives == () and baseline.compilation.mj_model.nu == 1
    assert prepared.follower_drives[0][-2:] == (0.,0.)
    assert prepared.scalar_controls == baseline.scalar_controls
    assert prepared.target_bindings == baseline.target_bindings
    for field in ('body_mass','body_inertia','jnt_range','dof_damping','eq_data'):
        np.testing.assert_array_equal(getattr(prepared.compilation.mj_model,field),getattr(baseline.compilation.mj_model,field))


def test_explicit_limits_and_mimic_rating_do_not_create_another_actuator(tmp_path):
    value = asset(tmp_path)
    rows = (replace(value.joints[0], lower_limit=-.8, upper_limit=.9, velocity_limit=.4),
            NativeJointInput("follower", lower_limit=-1., upper_limit=1., effort_limit=10,
                             velocity_limit=.6))
    result = compile_asset(replace(value, joints=rows))
    model = result.compilation.fresh_native_model()
    assert model.nu == 1 and model.neq == 1
    np.testing.assert_array_equal(model.jnt_range, [[-.8, .9], [-1., 1.]])
    assert result.joint_limits == (("arm::driver", -.8, .9, .4),
                                   ("arm::follower", -1., 1., .6))
    assert any("does not clamp velocity" in x for x in result.import_diagnostics)
    assert any("no actuator" in x for x in result.import_diagnostics)
    with pytest.raises(ValueError, match="no actuator"):
        compile_asset(replace(value, joints=(*value.joints,
                      NativeJointInput("follower", effort_limit=11))))


@pytest.mark.parametrize("kwargs", [dict(lower_limit=1., upper_limit=0.),
    dict(lower_limit=0., upper_limit=0.), dict(velocity_limit=-1.),
    dict(velocity_limit=float("nan")), dict(upper_limit=float("inf"))])
def test_invalid_joint_limit_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        NativeJointInput("j", **kwargs)


def test_partial_limits_preserve_other_bound_and_unlimited_requires_pair(tmp_path):
    value = asset(tmp_path)
    result = compile_asset(replace(value, joints=(replace(value.joints[0], upper_limit=.8),)))
    np.testing.assert_array_equal(result.compilation.mj_model.jnt_range[0], [-2, .8])
    path = Path(value.path)
    path.write_text(path.read_text().replace('type="prismatic"', 'type="continuous"')
                    .replace('lower="-2" upper="2" ', ''))
    with pytest.raises(ValueError, match="both finite"):
        compile_asset(replace(value, joints=(replace(value.joints[0], upper_limit=.8),)))


def test_revolute_limits_keep_radian_units(tmp_path):
    value = asset(tmp_path)
    path = Path(value.path)
    path.write_text(path.read_text().replace('type="prismatic"', 'type="revolute"'))
    result = compile_asset(replace(value, joints=(replace(value.joints[0],
                          lower_limit=-.4, upper_limit=.7),)))
    np.testing.assert_array_equal(result.compilation.mj_model.jnt_range[0], [-.4, .7])


def test_panda_original_dynamics_and_mimic_preserved():
    import phymeo.builders

    path = Path(phymeo.builders.__file__).parent / "panda_assets/panda.urdf"
    source = mj.MjModel.from_xml_path(str(path))
    result = compile_asset(NativeAssetInstance("panda", str(path), drive_mode="scalar_pd"))
    model = result.compilation.fresh_native_model()
    assert model.nq == 9 and model.nu == 8 and model.neq == 1
    assert any('initial_position_outside_limit: panda::panda_joint4' in row for row in result.import_diagnostics)
    assert result.initial_state.fields[1][1][3] == 0
    for i in range(source.njnt):
        j = model.joint("panda::" + source.joint(i).name)
        np.testing.assert_array_equal(
            model.dof_damping[j.dofadr],
            source.dof_damping[source.jnt_dofadr[i] : source.jnt_dofadr[i] + 1],
        )
    assert result.target_bindings[-1][1:] == result.target_bindings[-2][1:]


def test_mimic_names_are_scoped_per_instance(tmp_path):
    first = asset(tmp_path)
    result = compile_native_template(
        (first, replace(first, name="second")), timestep=0.002, gravity=(0, 0, 0)
    )
    model = result.compilation.fresh_native_model()
    assert model.neq == 2 and model.nu == 2
    for i in range(2):
        a, b = model.joint(model.eq_obj1id[i]).name, model.joint(model.eq_obj2id[i]).name
        assert a.split("::")[0] == b.split("::")[0]


def test_chain_and_rejection_boundaries(tmp_path):
    from copy import deepcopy
    from xml.etree import ElementTree as ET

    original = asset(tmp_path)
    path = Path(original.path)
    root = ET.fromstring(path.read_text())
    link = deepcopy(root.find("link[@name='b']"))
    link.set("name", "c")
    joint = deepcopy(root.find("joint[@name='follower']"))
    joint.set("name", "second")
    joint.find("child").set("link", "c")
    joint.find("mimic").attrib.update(joint="follower", multiplier="-.5", offset=".2")
    root.extend((link, joint))
    path.write_text(ET.tostring(root, encoding="unicode"))
    compiled = compile_asset(original)
    assert compiled.target_bindings[-1][:3] == ("arm::second", 0, 1.0)
    # Check arithmetic with a declared cold-conversion tolerance, not a trajectory threshold.
    assert compiled.target_bindings[-1][2] == 1
    np.testing.assert_allclose(compiled.target_bindings[-1][3], 0.15, atol=1e-12, rtol=0)
    data = mj.MjData(compiled.compilation.mj_model)
    compiled.initial_state.apply(compiled.compilation, data)
    np.testing.assert_allclose(data.qpos, [0.2, -0.3, 0.35], atol=1e-12, rtol=0)
    driven = compile_asset(replace(original, prepare_follower_drives=True,
        joints=(*original.joints, NativeJointInput("second", kp=6, kd=2))))
    assert len(driven.scalar_controls) == 1
    assert driven.follower_drives[-1][:3] == ("arm::second", 2, 0)
    np.testing.assert_allclose(driven.follower_drives[-1][3:], [1, .15, 6, 2], atol=1e-12, rtol=0)
    driven_data = mj.MjData(driven.compilation.mj_model)
    driven.initial_state.apply(driven.compilation, driven_data)
    np.testing.assert_allclose(driven_data.ctrl, [.8, 0, 2.1], atol=1e-12, rtol=0)
    for driver in ("missing", "second"):
        damaged = deepcopy(root)
        if driver == "second":
            ET.SubElement(damaged.find("joint[@name='driver']"), "mimic", joint=driver)
        else:
            damaged.find("joint[@name='follower']/mimic").set("joint", driver)
        path.write_text(ET.tostring(damaged, encoding="unicode"))
        with pytest.raises(ValueError, match="mimic"):
            compile_asset(original)

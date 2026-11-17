"""Complete-scene preservation is independent of articulation lowering."""

from dataclasses import replace
from hashlib import sha256

import mujoco
import numpy as np
import pytest

from phymeo.backends.physolve_initial import INTEGRATION_FIELDS, NativeInitialState
from phymeo.builders.native_scene import compile_native_scene

SCENE = """<mujoco model="original_scene"><option timestep=".002" cone="elliptic" gravity="0 0 -9.81"/>
<worldbody><geom name="floor" type="plane" size="1 1 .1"/>
<body name="arm"><joint name="hinge"/><geom size=".1"/><site name="tip"/></body>
<body name="object" pos="0 0 1"><freejoint name="root"/><geom size=".05"/></body>
</worldbody><actuator><motor name="motor" joint="hinge" gear="2" ctrlrange="-3 3"/></actuator>
<sensor><force name="force" site="tip"/><torque name="torque" site="tip"/></sensor></mujoco>"""


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "scene.xml"
    path.write_text(SCENE)
    return path


def test_complete_model_actuator_and_identity_preserved(source):
    template = compile_native_scene(source, name="task", capture_visuals=True)
    model = template.compilation.fresh_native_model()
    reference = mujoco.MjModel.from_xml_path(str(source))
    assert model.opt.timestep == reference.opt.timestep
    assert model.opt.cone == reference.opt.cone
    assert model.nsensor == 2
    for name in (
        "qpos0",
        "body_mass",
        "geom_bodyid",
        "geom_condim",
        "actuator_gear",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
    ):
        np.testing.assert_array_equal(getattr(model, name), getattr(reference, name))
    assert model.actuator(0).name == "motor"
    assert template.actuator_bindings == (("task::motor", 0),)
    assert template.scalar_controls == ()
    assert template.asset_identities[0].model_name == "original_scene"
    assert template.joint_dimensions == ((1, 1), (7, 6))
    assert len(template.scalar_joint_bindings) == 1
    assert len(template.shape_bindings) == reference.ngeom
    assert dict(template.native_identity)["body"][0] == "task::world"
    assert len(template.compile_visual_scene().instances) == reference.ngeom
    for count in (1, 8, 64):
        assert template.bind(count).template is template
    with pytest.raises(ValueError, match="complete scene contract"):
        replace(template, complete_scene_contract_version=99).bind(1).create_runtime(payload=None)


def test_snapshot_identity_all_inputs_and_explicit_source_precision(source, tmp_path):
    template = compile_native_scene(source, name="task")
    model = template.compilation.fresh_native_model()
    data = mujoco.MjData(model)
    data.qpos[4] += 6e-9
    data.qvel[:] = 0.25
    data.ctrl[:] = 0.7
    strict = NativeInitialState.from_data(template.compilation, data)
    target = mujoco.MjData(model)
    with pytest.raises(ValueError, match="unit WXYZ"):
        strict.apply(template.compilation, target)
    np.testing.assert_array_equal(target.qvel, np.zeros(model.nv))
    snapshot = tmp_path / "initial.npz"
    values = {name: np.asarray(getattr(data, name)) for name in INTEGRATION_FIELDS}
    np.savez(
        snapshot,
        **values,
        source_sha256=sha256(source.read_bytes()).hexdigest(),
        model_sha256=template.compilation.source.model_sha256,
    )
    imported = compile_native_scene(source, name="task", initial_state_path=snapshot)
    imported.initial_state.apply(imported.compilation, target)
    for name in INTEGRATION_FIELDS:
        np.testing.assert_array_equal(getattr(target, name), values[name])
    np.savez(
        snapshot,
        **values,
        source_sha256="wrong",
        model_sha256=template.compilation.source.model_sha256,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        compile_native_scene(source, name="task", initial_state_path=snapshot)
    np.savez(
        snapshot,
        **values,
        source_sha256=sha256(source.read_bytes()).hexdigest(),
        model_sha256="wrong",
    )
    with pytest.raises(ValueError, match="compiled model identity"):
        compile_native_scene(source, name="task", initial_state_path=snapshot)
    with pytest.raises(ValueError, match="unknown initial quaternion"):
        replace(strict, quaternion_contract="unknown")
    values["qpos"] = values["qpos"].copy()
    values["qpos"][4] = 1.001
    invalid = NativeInitialState(
        template.compilation.source.model_sha256,
        tuple((key, values[key]) for key in INTEGRATION_FIELDS),
        "mujoco_cuda_fp32_v1",
    )
    previous = {key: np.array(getattr(target, key)) for key in INTEGRATION_FIELDS}
    with pytest.raises(ValueError, match="unit WXYZ"):
        invalid.apply(template.compilation, target)
    for key, value in previous.items():
        np.testing.assert_array_equal(getattr(target, key), value)


def test_unchanged_root_with_changed_include_rejects_old_snapshot(tmp_path):
    source = tmp_path / "scene.xml"
    dependency = tmp_path / "bodies.xml"
    source.write_text('<mujoco><include file="bodies.xml"/></mujoco>')
    dependency.write_text(
        '<mujoco><worldbody><body><freejoint/><geom size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    template = compile_native_scene(source, name="task")
    snapshot = tmp_path / "state.npz"
    np.savez(
        snapshot,
        **dict(template.initial_state.fields),
        source_sha256=sha256(source.read_bytes()).hexdigest(),
        model_sha256=template.compilation.source.model_sha256,
    )
    dependency.write_text(dependency.read_text().replace('mass="1"', 'mass="2"'))
    with pytest.raises(ValueError, match="compiled model identity"):
        compile_native_scene(source, name="task", initial_state_path=snapshot)

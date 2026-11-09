"""H1 bounded 17/29/32-joint device mechanics and source identity.

AUTHORING_ONLY / UNEXECUTED. The parent owns all project execution. Expected
matrices are authored prismatic-tree incidence formulas, not HOST product M/FK.
"""

import gzip
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends._resident_floating_tree import compile_floating_tree_world
from phymeo.core.rigid import RigidBodyModel
from phymeo.core.robot import RobotModel
from phymeo.dynamics.floating_robot import FloatingRobotModel, FloatingRobotState

H = 1 / 128


def _encode(value):
    if isinstance(value, np.ndarray):
        return {
            "shape": value.shape,
            "dtype": value.dtype.str,
            "bytes_hex": value.tobytes().hex(),
            "values": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if type(value) is object:
        return {"opaque_process_local_type": "builtins.object", "identity_serialized": False}
    return value


def _save(tmp_path, name, value):
    with gzip.open(tmp_path / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(_encode(value), stream, allow_nan=False)


def _children(n, *, branched=False):
    parents = np.array([-1 if i == 0 or not branched else (i - 1) // 2 for i in range(n)])
    offsets = np.tile([1 / 32, 0, 0], (n, 1)) if branched else np.zeros((n, 3))
    return RobotModel(
        parents,
        ("prismatic",) * n,
        offsets,
        np.tile([0.0, 0, 0, 1], (n, 1)),
        np.tile([0.0, 0, 1], (n, 1)),
        np.ones(n),
        np.zeros((n, 3)),
        np.full((n, 3), 0.25),
        np.zeros(3),
    )


def _fixed(n):
    model = _children(n, branched=True)
    program = ph.DrivenRobotProgram(
        model, kp=np.zeros(n), kd=np.zeros(n), effort_limit=np.full(n, 8.0), delay_steps=0
    )
    state = program.initial_state(ph.RobotState(np.zeros(n), np.zeros(n)))
    return model, program, ph.FixedRobotTree(program, state)


def _floating(n, *, actuators=False):
    kwargs = {}
    if actuators:
        kwargs = {
            "joint_armature": np.full(n, 1 / 8),
            "joint_damping": np.linspace(1 / 8, 3 / 8, n),
        }
    model = FloatingRobotModel(
        RigidBodyModel(np.array([0.25]), np.full((1, 3), 0.5), np.zeros(3)),
        np.zeros(3),
        _children(n),
        **kwargs,
    )
    velocity = np.r_[np.zeros(6), np.full(n, 1 / 8) if actuators else np.zeros(n)]
    state = FloatingRobotState(np.r_[np.zeros(3), [0.0, 0, 0, 1], np.zeros(n)], velocity)
    tree = ph.FloatingRobotTree(
        model, state, kp=np.zeros(n), kd=np.zeros(n), effort_limit=np.full(n, 8.0)
    )
    return model, state, tree


@pytest.mark.tyx
def test_h1_fixed17_binary_tree_mass_and_fk(tyx_mode, tmp_path):
    n = 17
    model, program, tree = _fixed(n)
    # Body i carries every ancestor's unit z translation. Unit body masses give J.T J.
    incidence = np.zeros((n, n))
    depths = np.zeros(n)
    for body in range(n):
        ancestor = body
        while ancestor >= 0:
            incidence[body, ancestor] = 1
            depths[body] += 1
            ancestor = int(model.parents[ancestor])
    expected_mass = incidence.T @ incidence
    effort = np.linspace(-1 / 16, 1 / 16, n)
    expected_a = np.linalg.solve(expected_mass, effort)
    command = program.inputs(
        np.zeros(n),
        np.zeros(n),
        feedforward=np.zeros(n),
        loads=ph.RobotInputs(effort, np.zeros((n, 3)), np.zeros((n, 3))),
    )
    with ph.TyxWorldBatch([(41, tree)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({41: command})
        assert batch.step(H) == {41: True}
        accepted = batch.snapshot()[41]
        geometry = batch.tree_kinematics()[41]
        diagnostic = batch.tree_diagnostics()[41]
        np.testing.assert_allclose(diagnostic.mass, expected_mass, rtol=5e-5, atol=5e-6)
        np.testing.assert_allclose(diagnostic.acceleration, expected_a, rtol=5e-5, atol=5e-6)
        np.testing.assert_allclose(accepted.velocity, H * expected_a, rtol=5e-5, atol=5e-6)
        np.testing.assert_allclose(geometry.link_position_m[:, 0], depths / 32, atol=5e-6)
        np.testing.assert_allclose(
            geometry.link_position_m[:, 2], incidence @ (H * H * expected_a), atol=5e-6
        )
        assert diagnostic.equation_residual <= 2e-5
        _save(
            tmp_path,
            "h1-fixed17",
            dict(
                mode=tyx_mode,
                model=model,
                command=command,
                mass_reference=expected_mass,
                accepted=accepted,
                geometry=geometry,
                diagnostic=diagnostic,
                stats=batch.stats(),
            ),
        )


@pytest.mark.tyx
@pytest.mark.parametrize(
    "n,actuators", ((29, True), (32, False)), ids=("g29-actuated", "bound32-zero")
)
def test_h1_floating_capacity_step_and_material_restore(n, actuators, tyx_mode, tmp_path):
    model, state, tree = _floating(n, actuators=actuators)
    d = n + 6
    mass = np.zeros((d, d))
    mass[:3, :3] = np.eye(3) * (4 + n)
    mass[3:6, 3:6] = np.eye(3) * (0.5 + n / 4)
    mass[6:, 6:] = np.eye(n) * (1 + (1 / 8 if actuators else 0))
    mass[2, 6:] = 1
    mass[6:, 2] = 1
    damping = np.r_[np.zeros(6), np.linspace(1 / 8, 3 / 8, n) if actuators else np.zeros(n)]
    effort = np.linspace(-1 / 32, 1 / 32, n) if actuators else np.zeros(n)
    rhs = np.r_[np.zeros(6), effort] - damping * state.v
    expected_a = np.linalg.solve(mass, rhs)
    expected_v = state.v + H * np.linalg.solve(mass + H * np.diag(damping), rhs)
    drive = np.column_stack((np.zeros(n), np.zeros(n), effort))
    with ph.TyxWorldBatch(
        [(41, tree), (99, ph.FreeSphere(1, 0.05, gravity=(0, 0, 0)))], tyx_mode, cache_root=tmp_path
    ) as batch:
        initial = batch.snapshot()
        batch.set_controls({41: tree.inputs(drive)})
        assert batch.step(H) == {41: True, 99: True}
        accepted = batch.snapshot()
        diagnostic = batch.tree_diagnostics([41])[41]
        geometry = batch.tree_kinematics([41])[41]
        assert geometry.joint_count == n and geometry.body_count == n + 1
        assert geometry.velocity_count == d and geometry.configuration_count == n + 7
        np.testing.assert_allclose(diagnostic.mass, mass, rtol=5e-5, atol=1e-5)
        np.testing.assert_allclose(diagnostic.bias, damping * state.v, rtol=5e-5, atol=5e-6)
        np.testing.assert_allclose(diagnostic.acceleration, expected_a, rtol=5e-5, atol=5e-6)
        np.testing.assert_allclose(accepted[41].velocity, expected_v, rtol=5e-5, atol=5e-6)
        np.testing.assert_allclose(accepted[41].position[:3], H * expected_v[:3], atol=5e-6)
        np.testing.assert_allclose(accepted[41].position[7:], H * expected_v[6:], atol=5e-6)
        assert diagnostic.equation_residual <= 2e-5
        energy = batch.summary()[41]["kinetic_energy_j"]
        np.testing.assert_allclose(
            energy, 0.5 * expected_v @ mass @ expected_v, rtol=5e-5, atol=5e-6
        )
        if actuators:
            assert np.max(abs(expected_a - (expected_v - state.v) / H)) > 1e-5
            with pytest.raises(ValueError):
                tree.model.joint_armature.flags.writeable = True
        dynamic = ph.CollisionKey(41, 101, 0)
        static = ph.CollisionKey(41, 201, 0)
        shapes = ph.CollisionModel(
            (
                ph.CollisionShape(dynamic, "sphere", (701,), radius=1 / 32),
                ph.CollisionShape(static, "sphere", (801,), radius=1 / 32),
            )
        )
        bindings = (
            ph.ResidentFloatingTreeBinding(
                dynamic, 41, n, np.array([[0.0, 0, 1 / 8]]), topology=accepted[41].topology
            ),
            ph.ResidentStaticBinding(static, ph.CollisionMotion(np.array([[1.0, 0, 1 / 8]]))),
        )
        with batch.prepare_collision(shapes, bindings) as broad:
            result = broad.query()
            row = result.bounds[result.shape_keys.index(dynamic)]
            center = np.array([0.0, 0, 1 / 8]) + geometry.link_position_m[n]
            assert np.all(row[:3] <= center - 1 / 32) and np.all(row[3:] >= center + 1 / 32)
            before_other = batch.snapshot([99])[99]
            batch.restore({41: initial[41]})
            assert batch.snapshot([99])[99] == before_other
            with pytest.raises(ValueError):
                broad.require_current(result)
            recovered = broad.query()
            assert int(recovered.ticks[0]) == 0
        _save(
            tmp_path,
            f"h1-floating-{n}",
            dict(
                mode=tyx_mode,
                model=model,
                initial=initial,
                expected_mass=mass,
                expected_a=expected_a,
                expected_v=expected_v,
                accepted=accepted,
                diagnostic=diagnostic,
                geometry=geometry,
                energy=energy,
                bounds=result,
                recovered=recovered,
                stats=batch.stats(),
            ),
        )


@pytest.mark.parametrize("kind", ("fixed", "floating"))
def test_h1_joint33_rejected_before_runtime(kind, tmp_path):
    with pytest.raises(ValueError, match="32") as caught:
        if kind == "fixed":
            _, _, tree = _fixed(33)
            ph.TyxWorldBatch([(41, tree)], "interpret", cache_root=tmp_path)
        else:
            _floating(33)
    _save(
        tmp_path,
        "h1-over-cap-" + kind,
        dict(
            kind=kind,
            declared_joints=33,
            exception=type(caught.value).__name__,
            message=str(caught.value),
        ),
    )


@pytest.mark.parametrize("term", ("joint_armature", "joint_damping"))
def test_h1_actuator_copy_token_and_checkpoint_binding(term, tmp_path):
    model, _, tree = _floating(1)
    explicit_zero = replace(model, joint_armature=np.zeros(1), joint_damping=np.zeros(1))
    changed = replace(model, **{term: np.array([1 / 4])})
    original = compile_floating_tree_world(tree)
    zeros = compile_floating_tree_world(replace(tree, model=explicit_zero))
    modified_tree = replace(tree, model=changed)
    modified = compile_floating_tree_world(modified_tree)
    assert original.model_token == zeros.model_token
    assert original.topology == zeros.topology
    assert original.model_token != modified.model_token
    assert original.topology != modified.topology
    np.testing.assert_array_equal(getattr(modified_tree.model, term), [1 / 4])
    np.testing.assert_array_equal(getattr(modified, term), [1 / 4])
    with pytest.raises(ValueError):
        getattr(modified_tree.model, term).flags.writeable = True
    checkpoint = original.checkpoint(
        41,
        original.initial_root,
        original.initial_state,
        original.initial_generalized,
        0,
        tree.inputs(np.zeros((1, 3))),
    )
    with pytest.raises(ValueError, match="topology/model"):
        modified.restore_row(checkpoint, 41)
    _save(
        tmp_path,
        "h1-actuator-identity-" + term,
        dict(term=term, original=original, modified=modified, checkpoint=checkpoint),
    )

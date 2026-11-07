"""C9 ordinary floating trees: frozen public tests, AUTHORING_UNEXECUTED.

Only old APIs are imported at collection. The first node establishes a physical
7x7 HOST premise before looking up the new resident public API. Execution belongs
to the parent; all literal masses, loads and tolerances are declared inputs.
"""

import gzip
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

import phymeo as ph
from phymeo.core.rigid import RigidBodyModel
from phymeo.core.robot import RobotModel
from phymeo.dynamics.floating_robot import FloatingRobotModel, FloatingRobotState
from phymeo.solvers.floating_tree import FloatingRobotInputs, floating_dynamics_terms

H = 1 / 1024
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4
WORLD = 503


def _json(value):
    if is_dataclass(value):
        return {field.name: _json(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def _save(directory, name, payload):
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(_json(payload), stream, allow_nan=False, separators=(",", ":"))


def _close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _p03_host():
    root = RigidBodyModel(np.array([0.5]), np.full((1, 3), 0.5), np.zeros(3))
    child = RobotModel(
        np.array([-1]), ("prismatic",), np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]),
        np.array([1.0]), np.zeros((1, 3)), np.full((1, 3), 0.25), np.zeros(3),
    )
    model = FloatingRobotModel(root, np.zeros(3), child)
    state = FloatingRobotState(np.array([0.0, 0, 0, 0, 0, 0, 1, 0]), np.zeros(7))
    return model, state


@pytest.mark.tyx
def test_p03_public_floating_tree_keeps_complete_mass_and_internal_recoil(tyx_mode, tmp_path):
    model, state = _p03_host()
    expected_mass = np.diag([3.0, 3, 3, 0.75, 0.75, 0.75, 1])
    expected_mass[0, 6] = expected_mass[6, 0] = 1
    expected_a = np.array([-1.5, 0, 0, 0, 0, 0, 4.5])
    internal = FloatingRobotInputs(np.array([3.0]), np.zeros((2, 3)), np.zeros((2, 3)))
    host = floating_dynamics_terms(model, state, internal)
    assert model.body_count == 2 and host.mass_matrix.shape == (7, 7)
    _close(host.mass_matrix, expected_mass, atol=1e-12, rtol=0)
    _close(host.bias, np.zeros(7), atol=1e-12, rtol=0)
    _close(host.applied, [0, 0, 0, 0, 0, 0, 3], atol=1e-12, rtol=0)
    _close(host.acceleration, expected_a, atol=1e-12, rtol=0)
    # Mxx=[[3,1],[1,1]] gives (-3/2,9/2), with child world ax=3.
    assert 2 * expected_a[0] + expected_a[0] + expected_a[6] == 0
    record = dict(
        case="P03", mode=tyx_mode, dt_s=H, model=model, initial=state,
        literal_mass=expected_mass, literal_acceleration=expected_a,
        host_premise=host, drive=[[0.0, 0.0, 3.0]], external=None,
    )
    try:
        from phymeo import backends

        tree = backends.FloatingRobotTree(
            model, state, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.array([8.0])
        )
        packet = tree.inputs(np.array([[0.0, 0.0, 3.0]]))
        with ph.TyxWorldBatch([(WORLD, tree)], tyx_mode, cache_root=tmp_path) as batch:
            batch.set_controls({WORLD: packet})
            entry = batch.snapshot()[WORLD]
            assert len(entry.position) == 8 and len(entry.velocity) == 7
            assert batch.step(H) == {WORLD: True}
            accepted = batch.snapshot()[WORLD]
            geometry = batch.tree_kinematics()[WORLD]
            diagnostic = batch.tree_diagnostics()[WORLD]
            record.update(
                entry=entry, accepted=accepted, geometry=geometry,
                diagnostic=diagnostic, energy=batch.summary(), stats=batch.stats(),
            )
            assert accepted.tick == entry.tick + 1
            _close(diagnostic.mass, expected_mass, atol=2e-5)
            _close(diagnostic.acceleration, expected_a)
            _close(diagnostic.drive, [3.0])
            _close(accepted.velocity, H * expected_a)
            _close(accepted.position[:3], [-1.5 * H * H, 0, 0])
            _close(accepted.position[3:7], [0, 0, 0, 1])
            _close(accepted.position[7:], [4.5 * H * H])
            _close(geometry.linear_velocity_m_s, [[-1.5 * H, 0, 0], [3 * H, 0, 0]])
            _close(geometry.angular_velocity_rad_s, np.zeros((2, 3)))
            _close(2 * geometry.linear_velocity_m_s[0] + geometry.linear_velocity_m_s[1], np.zeros(3), atol=1e-12, rtol=0)
            assert accepted.control == entry.control and not accepted.history
            assert (geometry.joint_count, geometry.body_count, geometry.velocity_count, geometry.configuration_count) == (1, 2, 7, 8)
            assert batch.stats()["floating_world_count"] == 1
            assert batch.stats()["submitted_kernels"] > 0
    except Exception as error:
        record["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        _save(tmp_path, "p03-public-floating", record)
"""C10 public floating contact inputs and independent physical anchors.

AUTHORING_UNEXECUTED. Parent alone executes these tests. The first node proves
an old HOST finite patch/law and a literal root-only mass matrix before calling
the already-public prepare_contacts entry whose floating domain is missing.
"""

import gzip
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from itertools import product

import numpy as np
import pytest

import phymeo as ph
from phymeo.core.rigid import RigidBodyModel
from phymeo.dynamics.floating_robot import FloatingRobotModel, FloatingRobotState
from phymeo.geometry.contact_patch_adapter import contact_patch_sample
from phymeo.solvers.contact_patch_loads import sample_patch_loads
from phymeo.solvers.floating_tree import FloatingRobotInputs, floating_dynamics_terms

H = 1 / 1024
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4
WORLD = 503


def _json(value):
    if is_dataclass(value):
        return {f.name: _json(getattr(value, f.name)) for f in fields(value)}
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


def _box(low, high):
    return np.array(list(product(*zip(low, high, strict=True))), dtype=float)


@pytest.mark.tyx
def test_f01_public_root_contact_keeps_full_six_coordinate_response(tyx_mode, tmp_path):
    root = RigidBodyModel(np.array([0.5]), np.full((1, 3), 0.5), np.zeros(3))
    model = FloatingRobotModel(root, np.zeros(3), None)
    initial = FloatingRobotState(np.array([0.0, 0, 3 / 32, 0, 0, 0, 1]), np.zeros(6))
    ball_key, wall_key = ph.CollisionKey(WORLD, 101, 7), ph.CollisionKey(WORLD, 900, 2)
    ball = ph.CollisionShape(ball_key, "sphere", (101,), 1 / 8, revision=7)
    wall = ph.CollisionShape(wall_key, "hull", tuple(501 + 3 * i for i in range(8)), revision=3)
    wall_points = _box([-1, -1, -1 / 4], [1, 1, 0])
    patch = contact_patch_sample(ball, wall, [[0, 0, 3 / 32]], wall_points)
    assert len(patch.sites) == 1 and patch.dimension == 0
    site = patch.sites[0]
    _close(site.witness.gap_m, -1 / 32, atol=1e-12, rtol=0)
    _close(site.witness.normal, [0, 0, -1], atol=1e-12, rtol=0)
    assert site.weight == 1
    law = ph.RobotContactMaterial(64, 0, 0)
    measured = sample_patch_loads(patch, law, np.zeros((1, 3)), np.zeros((1, 3)))
    _close(measured["first_force"], [[0, 0, 2]], atol=1e-12, rtol=0)
    _close(measured["second_force"], [[0, 0, -2]], atol=1e-12, rtol=0)
    _close(np.cross(np.array(site.load_point) - initial.q[:3], measured["first_force"][0]), 0, atol=1e-12, rtol=0)
    known = FloatingRobotInputs(np.zeros(0), np.array([[0.0, 0, 2]]), np.zeros((1, 3)))
    host = floating_dynamics_terms(model, initial, known)
    literal_mass = np.diag([2.0, 2, 2, 0.5, 0.5, 0.5])
    literal_acceleration = np.array([0.0, 0, 1, 0, 0, 0])
    _close(host.mass_matrix, literal_mass, atol=1e-12, rtol=0)
    _close(host.bias, np.zeros(6), atol=1e-12, rtol=0)
    _close(host.applied, [0, 0, 2, 0, 0, 0], atol=1e-12, rtol=0)
    _close(host.acceleration, literal_acceleration, atol=1e-12, rtol=0)
    record = dict(
        case="F01-public", mode=tyx_mode, dt_s=H, model=model, initial=initial,
        shapes=(ball, wall), wall_points=wall_points, local_ball=np.zeros((1, 3)),
        law=law, host_patch=patch, host_loads=measured, host_terms=host,
        literal_mass=literal_mass, literal_acceleration=literal_acceleration,
        drive=np.zeros((0, 3)), external=None,
    )
    try:
        tree = ph.FloatingRobotTree(model, initial, kp=np.zeros(0), kd=np.zeros(0), effort_limit=np.zeros(0))
        with ph.TyxWorldBatch([(WORLD, tree)], tyx_mode, cache_root=tmp_path) as batch:
            batch.set_controls({WORLD: tree.inputs(np.zeros((0, 3)))})
            before = batch.snapshot()[WORLD]
            before_geometry = batch.tree_kinematics()[WORLD]
            assert (before_geometry.body_count, before_geometry.velocity_count) == (1, 6)
            _close(before_geometry.position_m, [[0, 0, 3 / 32]])
            collision = ph.CollisionModel((ball, wall))
            bindings = (
                ph.ResidentFloatingTreeBinding(ball_key, WORLD, 0, np.zeros((1, 3)), topology=before.topology),
                ph.ResidentStaticBinding(wall_key, ph.CollisionMotion(wall_points)),
            )
            record.update(entry=before, entry_geometry=before_geometry, bindings=bindings)
            # Effective RED: all physical and original resident prerequisites
            # above must pass before the current floating-domain guard is met.
            try:
                with batch.prepare_contacts(collision, bindings, {(ball_key, wall_key): law}) as plan:
                    assert batch.step_contacts(plan, H) == {WORLD: True}
                    receipt = plan.last_accepted[WORLD]
                    after = batch.snapshot()[WORLD]
                    geometry = batch.tree_kinematics()[WORLD]
                    record.update(receipt=receipt, accepted=after, geometry=geometry, plan_stats=plan.stats(), batch_stats=batch.stats())
                    assert receipt.entry_state == before and receipt.accepted_state == after
                    assert (receipt.entry_tick, receipt.exit_tick) == (0, 1)
                    assert after.control == before.control and not after.history
                    assert receipt.selected_drive.shape == (0, 3) and receipt.actual_drive.shape == (0,)
                    assert receipt.law_wrench_fp64.shape == (1, 6)
                    _close(receipt.law_wrench_fp64, [[0, 0, 2, 0, 0, 0]])
                    _close(receipt.input_external, np.zeros(6))
                    _close(receipt.contact_generalized, [0, 0, 2, 0, 0, 0])
                    _close(receipt.static_reactions[wall_key].force_n, [0, 0, -2])
                    _close(receipt.static_reactions[wall_key].torque_nm, np.zeros(3))
                    d = receipt.diagnostics
                    assert d["coordinate_kind"] == "floating"
                    assert tuple(d[k] for k in ("joint_count", "body_count", "velocity_count", "position_count")) == (0, 1, 6, 7)
                    _close(d["mass"], literal_mass, atol=2e-5)
                    _close(d["acceleration"], literal_acceleration)
                    _close(after.velocity, H * literal_acceleration)
                    _close(after.position[:3], [0, 0, 3 / 32 + H * H])
                    _close(after.position[3:], [0, 0, 0, 1])
                    _close(geometry.linear_velocity_m_s, [[0, 0, H]])
                    assert d["floating_diagnostic"].equation_residual == d["equation_residual"]
                    assert batch.stats()["floating_world_count"] == 1
                    assert batch.stats()["submitted_kernels"] > 0
            except Exception:
                record.update(retained=batch.snapshot(), retained_geometry=batch.tree_kinematics(), batch_stats=batch.stats())
                raise
    except Exception as error:
        record["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        _save(tmp_path, "f01-public-floating-contact", record)

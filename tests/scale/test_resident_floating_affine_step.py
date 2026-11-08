"""C11 floating-affine public cases; AUTHORING_UNEXECUTED.

Existing HOST mechanics establish the literal premise before the first test
looks up the new public device declaration. No new product import at collection.
"""

import gzip
import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.dynamics.floating_robot import FloatingRobotState
from phymeo.solvers.floating_tree import FloatingRobotInputs, floating_dynamics_terms

H = 1 / 1024
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4
WORLD = 503
ASSETS = Path(__file__).resolve().parents[1] / "fixtures" / "urdf"
JAWS = ASSETS / "floating_mimic_jaws.urdf"
ELBOW = ASSETS / "floating_mimic_elbow.urdf"
RAW_LIMIT = 256 * 1024 * 1024
COMPRESSED_LIMIT = 64 * 1024 * 1024


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def _save(directory, name, payload):
    """Complete lossless output; resource exhaustion is an IO failure, not physics."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".json.gz")
    size = 0
    encoder = json.JSONEncoder(allow_nan=False, separators=(",", ":"))
    with gzip.open(path, "wb") as stream:
        for piece in encoder.iterencode(_json(payload)):
            data = piece.encode("utf-8")
            size += len(data)
            if size > RAW_LIMIT:
                raise RuntimeError("C11 complete payload exceeds declared 256 MiB raw limit")
            stream.write(data)
    if path.stat().st_size > COMPRESSED_LIMIT:
        raise RuntimeError("C11 complete payload exceeds declared 64 MiB gzip limit")


def _close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _jaws(*, gravity=(0.0, 0.0, 0.0)):
    return ph.load_floating_mimic_urdf(JAWS, gravity=gravity)


def _jaws_state():
    return FloatingRobotState(np.array([0.0, 0, 0, 0, 0, 0, 1, 0]), np.zeros(7))


def _fixture_record(path):
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "utf8": raw.decode("utf-8")}


@pytest.mark.tyx
def test_a01_public_affine_floating_root_reaction(tyx_mode, tmp_path):
    asset, initial = _jaws(), _jaws_state()
    full = asset.full_state(initial)
    external = FloatingRobotInputs.zeros(2)
    host = floating_dynamics_terms(asset.tree.model, full, external)
    literal_x_mass = np.array([[5.0, 1, 2], [1, 1, 0], [2, 0, 2]])
    literal_reduced_x_mass = np.array([[5.0, -1], [-1, 3]])
    literal_a = np.array([1 / 14, 0, 0, 0, 0, 0, 5 / 14, -5 / 14])
    literal_constraint = np.array([0.0, 0, 0, 0, 0, 0, -4 / 7, -4 / 7])
    assert asset.tree.model.body_count == 3
    assert full.q.shape == (9,) and full.v.shape == (8,)
    assert initial.q.shape == (8,) and initial.v.shape == (7,)
    assert host.mass_matrix.shape == (8, 8)
    _close(host.mass_matrix[np.ix_([0, 6, 7], [0, 6, 7])], literal_x_mass,
           atol=2e-12, rtol=0)
    _close(host.bias, np.zeros(8), atol=2e-12, rtol=0)
    _close(asset.matrix, [[1.0], [-1.0]], atol=0, rtol=0)
    _close(asset.offset_vector, [0, 0.004], atol=0, rtol=0)
    _close(asset.constraint_rows, [[1.0, 1.0]], atol=0, rtol=0)
    host_step = asset.step(initial, effort=np.array([1.0]), dt_s=H)
    _close(host_step.loads.acceleration_full, literal_a, atol=2e-12, rtol=0)
    _close(host_step.loads.constraint_effort_full, literal_constraint, atol=2e-12, rtol=0)
    assert host_step.loads.multiplier_force == pytest.approx(-4 / 7, abs=2e-12)
    assert host_step.loads.constraint_power_W == pytest.approx(0.0, abs=2e-12)
    # Independent momentum identity: 2 ar + (ar+u'') + 2(ar-u'') = 0.
    assert 5 * (1 / 14) - 5 / 14 == 0
    record = dict(case="A01", mode=tyx_mode, dt_s=H, asset=asset,
                  fixture=_fixture_record(JAWS), initial=initial, full_initial=full,
                  external=external, drive=[[0.0, 0.0, 1.0]],
                  literal_x_mass=literal_x_mass, literal_reduced_x_mass=literal_reduced_x_mass,
                  literal_acceleration=literal_a, literal_constraint=literal_constraint,
                  host_premise=host, host_step=host_step, frames=[])
    try:
        # This lookup is deliberately after the real old-API physical premises.
        tree = ph.FloatingAffineRobotTree(
            asset, initial, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.array([1.5])
        )
        packet = tree.inputs(np.array([[0.0, 0.0, 1.0]]), external=external)
        record["packet"] = packet
        with ph.TyxWorldBatch([(WORLD, tree)], tyx_mode, cache_root=tmp_path / "cache") as batch:
            batch.set_controls({WORLD: packet})
            entry = batch.snapshot()[WORLD]
            assert len(entry.position) == 8 and len(entry.velocity) == 7
            assert len(entry.full_position) == 9 and len(entry.full_velocity) == 8
            assert len(entry.control) == 5 and entry.control[0] == packet.graph_token
            assert batch.step(H) == {WORLD: True}
            accepted = batch.snapshot()[WORLD]
            geometry = batch.tree_kinematics()[WORLD]
            diagnostic = batch.tree_diagnostics()[WORLD]
            record["frames"].append(dict(entry=entry, accepted=accepted, geometry=geometry,
                                         diagnostic=diagnostic, summary=batch.summary(),
                                         stats=batch.stats()))
            assert accepted.tick == entry.tick + 1
            assert diagnostic.mass.shape == (8, 8)
            assert diagnostic.reduced_mass.shape == (7, 7)
            _close(diagnostic.mass[np.ix_([0, 6, 7], [0, 6, 7])], literal_x_mass, atol=2e-5)
            _close(diagnostic.reduced_mass[np.ix_([0, 6], [0, 6])],
                   literal_reduced_x_mass, atol=2e-5)
            _close(diagnostic.full_acceleration, literal_a)
            _close(diagnostic.acceleration, literal_a[:7])
            _close(diagnostic.drive, [1.0])
            _close(diagnostic.constraint_effort_full, literal_constraint, atol=2e-5)
            _close(diagnostic.root_constraint_residual, np.zeros(6), atol=2e-5, rtol=0)
            _close(diagnostic.row_multipliers, [-4 / 7], atol=2e-5)
            assert diagnostic.constraint_power_w == pytest.approx(0.0, abs=2e-5)
            _close(accepted.velocity, H * literal_a[:7])
            _close(accepted.full_velocity, H * literal_a)
            _close(accepted.position[:3], [H * H / 14, 0, 0])
            _close(accepted.position[7:], [H * H * 5 / 14])
            _close(accepted.full_position[7:], [H * H * 5 / 14, 0.004 - H * H * 5 / 14])
            _close(accepted.applied_effort, [1.0])
            _close(accepted.full_applied_effort, [1.0, 0.0])
            _close(accepted.generalized_applied, [0, 0, 0, 0, 0, 0, 1])
            _close(accepted.full_generalized_applied, [0, 0, 0, 0, 0, 0, 1, 0])
            _close(geometry.position, accepted.full_position)
            _close(geometry.velocity, accepted.full_velocity)
            _close(geometry.independent_position, accepted.position)
            _close(geometry.independent_velocity, accepted.velocity)
            assert geometry.joint_names == ("driver",)
            assert geometry.full_joint_names == ("driver", "follower")
            v = np.asarray(accepted.full_velocity)
            _close((5 * v[0] + v[6] + 2 * v[7]) / 5, 0.0, atol=ATOL, rtol=0)
            for value in (diagnostic.mass, diagnostic.reduced_mass, diagnostic.full_acceleration,
                          diagnostic.constraint_effort_full, geometry.position):
                assert not value.flags.writeable
            record["retained"] = batch.snapshot()
    except BaseException as exc:
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        _save(tmp_path, "a01-public", record)
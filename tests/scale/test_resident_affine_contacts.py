"""C7 frozen affine resident-contact cases; authoring only, UNEXECUTED.

The existing C5 reference is an independent full-component/KKT oracle. Product
M/J/loads never define expected dynamics. All execution belongs to the parent.
"""

import copy
import gzip
import importlib.util
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.geometry.contact_patch_adapter import contact_patch_sample
from phymeo.solvers.contact_patch_loads import sample_patch_loads

H = 1 / 1024
ATOL = 5e-6
RTOL = 5e-5
WINDOW_ATOL = 2e-4
WORLD = 503


@lru_cache(maxsize=1)
def _ref():
    path = Path(__file__).resolve().parents[1] / "reference_newton/robot/resident_affine_reference.py"
    spec = importlib.util.spec_from_file_location("c7_frozen_independent_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    with gzip.open(directory / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(_json(payload), stream, allow_nan=False, separators=(",", ":"))


def _close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _box(low, high):
    return np.array(list(product(*zip(low, high, strict=True))), dtype=float)


def _case(name):
    data = copy.deepcopy(_ref().case(name))
    # Replace the original imposed 3 N by actual finite contact. Never add both.
    if name in ("A01F", "A05G"):
        data["force"][:] = 0
    return data


def _asset(directory, data, *, reorder=False):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "mechanism.urdf"
    path.write_text(_ref().urdf(data, reorder=reorder), encoding="utf-8")
    return ph.load_mimic_urdf(path, gravity=data["model"]["gravity"])


def _orders(asset, data):
    names = tuple(data["model"]["joint_names"][i] for i in _ref().coordinates(data["model"]))
    independent = tuple(names[i] for i in data["independent"])
    return [names.index(name) for name in asset.tree.joint_names], [independent.index(name) for name in asset.joint_names]


def _world(asset, data, *, kp=None, kd=None, cap=None, delay=0, iterations=512):
    _, order = _orders(asset, data)
    k = len(order)
    def vector(value, default):
        return np.full(k, default) if value is None else np.asarray(value, dtype=float)[order]
    return ph.AffineRobotTree(
        asset, ph.RobotState(np.asarray(data["s"], float)[order], np.asarray(data["u"], float)[order]),
        kp=vector(kp, 0), kd=vector(kd, 0), effort_limit=vector(cap, 16),
        delay_steps=delay, control_max_iterations=iterations,
    )


def _control(world, asset, data, *, target=None, rate=None, ff=None):
    full_order, order = _orders(asset, data)
    active = list(_ref().coordinates(data["model"]))
    def vector(value, default):
        return np.asarray(default if value is None else value, dtype=float)[order]
    return world.inputs(
        vector(target, data["s"]), vector(rate, np.zeros(len(order))),
        feedforward=vector(ff, data["motor"]),
        loads=ph.RobotInputs(
            np.asarray(data["effort"], float)[full_order],
            np.asarray(data["force"], float)[active][full_order],
            np.asarray(data["torque"], float)[active][full_order],
        ),
    )


def _a01_geometry(batch, asset, *, world=WORLD):
    group = asset.tree.sites["link1"].link_index
    moving, fixed = ph.CollisionKey(world, 100 + group, 11), ph.CollisionKey(world, 900, 1)
    first = ph.CollisionShape(moving, "sphere", (71,), radius=1 / 8)
    second = ph.CollisionShape(fixed, "hull", tuple(range(201, 209)))
    local = np.zeros((1, 3))
    support = _box((-13 / 64, 1 / 4, -1 / 4), (3 / 64, 3 / 4, 1 / 4))
    checkpoint = batch.snapshot([world])[world]
    bindings = (
        ph.ResidentAffineTreeBinding(moving, world, group, local, topology=checkpoint.topology),
        ph.ResidentStaticBinding(fixed, ph.CollisionMotion(support)),
    )
    return ph.CollisionModel((first, second)), bindings, (moving, fixed), local, support


@pytest.mark.tyx
def test_c701_public_affine_contact_replaces_follower_three_newtons(tyx_mode, tmp_path):
    data = _case("A01F")
    asset = _asset(tmp_path, data)
    assert asset.tree.model.dof_count == 2 and len(asset.joint_names) == 1
    np.testing.assert_array_equal(asset.matrix, ((1,), (2,)))
    np.testing.assert_array_equal(asset.offset_vector, (0, 1 / 8))
    np.testing.assert_array_equal(asset.full_state(ph.RobotState(np.zeros(1), np.zeros(1))).q, (0, 1 / 8))
    np.testing.assert_array_equal(data["force"], 0)
    world = _world(asset, data)
    record = dict(input=data, h_s=H, backend=tyx_mode)
    with ph.TyxWorldBatch([(WORLD, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: _control(world, asset, data)})
        before = batch.snapshot()
        model, bindings, pair, local, support = _a01_geometry(batch, asset)
        geometry = batch.tree_kinematics([WORLD])[WORLD]
        group = bindings[0].group_index
        _close(geometry.com_position_m[group], (1 / 8, 1 / 2, 0))
        points = geometry.link_position_m[group] + local @ geometry.link_rotation[group].T
        law = ph.RobotContactMaterial(64, 0, 0)
        patch = contact_patch_sample(model.shapes[0], model.shapes[1], points, support)
        assert len(patch.sites) == 1 and patch.sites[0].weight == 1
        assert patch.sites[0].witness.gap_m == pytest.approx(-3 / 64, abs=1e-12)
        sampled = sample_patch_loads(patch, law, np.zeros((1, 3)), np.zeros((1, 3)))
        _close(sampled["first_force"], ((3, 0, 0),), atol=1e-12, rtol=0)
        with batch.prepare_collision(model, bindings) as collision:
            result = collision.query()
            assert result.candidates == (pair,)
            assert batch.snapshot() == before
        record.update(before=before, geometry=geometry, collision_model=model, bindings=bindings, original_patch=patch, original_force=sampled)
        try:
            # The accepted asset/full FK/binding and actual old 3 N law precede
            # the real missing capability: ordinary-only prepare_contacts.
            with batch.prepare_contacts(model, bindings, {pair: law}) as plan:
                assert batch.step_contacts(plan, H) == {WORLD: True}
                after = batch.snapshot()[WORLD]
                diagnostic = batch.tree_diagnostics([WORLD])[WORLD]
                receipt = plan.last_accepted[WORLD]
                record.update(after=after, diagnostic=diagnostic, receipt=receipt)
                _close(after.velocity, (H,))
                _close(after.position, (H * H,))
                _close(after.full_velocity, (H, 2 * H))
                _close(receipt.input_external, (0, 0))
                _close(receipt.contact_generalized, (0, 3))
                _close(diagnostic.reduced_mass, ((6,),))
                _close(diagnostic.reduced_external, (6,))
                _close(diagnostic.acceleration, (1,))
                _close(diagnostic.full_acceleration, (1, 2))
                _close(diagnostic.row_multipliers, (-1,))
                _close(diagnostic.constraint_effort_full, (2, -1))
                assert after.control == before[WORLD].control
        except Exception as error:
            record["error"] = dict(type=type(error).__name__, message=str(error))
            raise
        finally:
            record["retained"] = batch.snapshot()
            record["stats"] = batch.stats()
            _save(tmp_path, "c701-public-contact", record)

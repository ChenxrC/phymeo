"""C8 same-tree double-dynamic contact, tests-first / UNEXECUTED.

The parent owns every project execution. Full physical inputs and independent
oracles remain separate from the actual device geometry and contact receipts.
"""

import gzip
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

import phymeo as ph
from phymeo.geometry.contact_patch_adapter import contact_patch_sample
from phymeo.solvers.contact_patch_loads import sample_patch_loads

H = 1 / 1024
WORLD = 503
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4


def _json(value):
    if is_dataclass(value):
        return {f.name: _json(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def _save(directory, name, payload):
    with gzip.open(directory / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(_json(payload), stream, allow_nan=False, separators=(",", ":"))


def _close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _d01(*, velocity=(0, 0), delay=0, kp=0, kd=0, cap=8):
    model = ph.RobotModel(
        np.array([-1, -1]), ("prismatic", "prismatic"),
        np.array([[0, 0, 0], [7 / 32, 0, 0]]),
        np.tile([0., 0., 0., 1.], (2, 1)), np.tile([1., 0., 0.], (2, 1)),
        np.array([2., 1.]), np.zeros((2, 3)), np.full((2, 3), 1 / 8), np.zeros(3),
    )
    program = ph.DrivenRobotProgram(
        model, kp=np.full(2, kp), kd=np.full(2, kd),
        effort_limit=np.full(2, cap), delay_steps=delay,
    )
    state = program.initial_state(ph.RobotState(np.zeros(2), np.array(velocity, float)))
    world = ph.FixedRobotTree(program, state, control_max_iterations=512)
    command = program.inputs(np.zeros(2), np.zeros(2), feedforward=np.zeros(2), loads=ph.RobotInputs.zeros(2))
    return program, world, command


def _two_spheres(batch, *, world=WORLD, groups=(0, 1), local=None, reverse=False):
    local = (np.zeros((1, 3)), np.zeros((1, 3))) if local is None else local
    keys = (ph.CollisionKey(world, 101, 7), ph.CollisionKey(world, 309, 2))
    shapes = tuple(ph.CollisionShape(k, "sphere", (11 + 19 * i,), radius=1 / 8) for i, k in enumerate(keys))
    topology = batch.snapshot([world])[world].topology
    bindings = tuple(ph.ResidentTreeBinding(k, world, g, p, topology=topology) for k, g, p in zip(keys, groups, local, strict=True))
    if reverse:
        shapes, bindings = tuple(reversed(shapes)), tuple(reversed(bindings))
    return ph.CollisionModel(shapes), bindings, keys


@pytest.mark.tyx
def test_d01_public_double_dynamic_tree_follows_two_sided_two_newton_patch(tyx_mode, tmp_path):
    """The valid RED must reach the new opt-in, after old HOST/device premises."""
    program, world, command = _d01()
    law = ph.RobotContactMaterial(64, 0, 0)
    record = dict(h_s=H, mode=tyx_mode, physical_model=program.model, initial=world.state, command=command, law=law,
                  independent=dict(mass=[[2, 0], [0, 1]], gap=-1 / 32, force=[[-2, 0, 0], [2, 0, 0]], acceleration=[-1, 2]))
    with ph.TyxWorldBatch([(WORLD, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: command})
        before, geometry = batch.snapshot()[WORLD], batch.tree_kinematics()[WORLD]
        model, bindings, pair = _two_spheres(batch)
        points = (np.array([[0., 0., 0.]]), np.array([[7 / 32, 0., 0.]]))
        assert tuple(program.model.parents) == (-1, -1)
        assert before.position == before.velocity == (0., 0.)
        _close(geometry.link_position_m, [[0, 0, 0], [7 / 32, 0, 0]], atol=1e-12, rtol=0)
        patch = contact_patch_sample(model.shapes[0], model.shapes[1], *points)
        assert patch is not None and len(patch.sites) == 1
        assert patch.sites[0].witness.gap_m == -1 / 32
        assert sum(s.weight for s in patch.sites) == 1
        sampled = sample_patch_loads(patch, law, np.zeros((1, 3)), np.zeros((1, 3)))
        _close(sampled["first_force"], [[-2, 0, 0]], atol=1e-12, rtol=0)
        _close(sampled["second_force"], [[2, 0, 0]], atol=1e-12, rtol=0)
        record.update(before=before, geometry=geometry, collision_model=model, bindings=bindings, host_patch=patch, host_loads=sampled)
        try:
            # No new HOST bridge is called before this public capability boundary.
            with batch.prepare_contacts(model, bindings, {pair: law}, tree_dynamic_pairs=True) as plan:
                assert batch.step_contacts(plan, H) == {WORLD: True}
                receipt = plan.last_accepted[WORLD]
                record.update(receipt=receipt)
                after = batch.snapshot()[WORLD]
                assert after.tick == before.tick + 1
                _close(after.velocity, np.array([-1., 2.]) * H)
                _close(after.position, np.array([-1., 2.]) * H * H)
                _close(receipt.contact_generalized, [-2, 2])
                _close(receipt.law_wrench_fp64, [[-2, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]])
                _close(receipt.diagnostics["mass"], np.diag([2., 1.]), atol=2e-5)
                _close(receipt.diagnostics["acceleration"], [-1, 2])
                assert receipt.diagnostics["tree_dynamic_pairs"] is True
                assert not receipt.static_reactions
                assert after.control == before.control
                _close(2 * after.velocity[0] + after.velocity[1], 0, atol=1e-12, rtol=0)
                assert batch.stats()["submitted_kernels"] > 0
        finally:
            record.update(retained=batch.snapshot(), stats=batch.stats())
            _save(tmp_path, "d01-public-double-dynamic", record)

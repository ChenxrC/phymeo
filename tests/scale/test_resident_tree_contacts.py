"""C6 frozen entry-contact feedback cases; tests-first authoring, UNEXECUTED.

Current-pose contact is sampled once from accepted device geometry. Independent
oracles below never use product dynamics, FK, or contact output as their truth.
Execution and all backend/static checks belong to the parent serial owner.
"""

import gzip
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from itertools import product

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends import tyx_world_batch as resident
from phymeo.geometry.contact_patch_adapter import contact_patch_sample
from phymeo.solvers.contact_patch_loads import sample_patch_loads

H = 1 / 1024
RTOL = 5e-5
ATOL = 5e-6
WINDOW_ATOL = 2e-4


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


def _save(path, name, value):
    with gzip.open(path / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(_json(value), stream, indent=2, allow_nan=False)


def _box(lo, hi):
    return np.asarray(list(product(*zip(lo, hi, strict=True))), dtype=float)


def _close(actual, expected, *, atol=ATOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=RTOL)


def _program(model, *, q=None, v=None, kp=None, kd=None, delay=0):
    n = model.dof_count
    program = ph.DrivenRobotProgram(
        model,
        kp=np.zeros(n) if kp is None else np.asarray(kp, dtype=float),
        kd=np.zeros(n) if kd is None else np.asarray(kd, dtype=float),
        effort_limit=np.full(n, 8.0),
        delay_steps=delay,
    )
    initial = program.initial_state(
        ph.RobotState(
            np.zeros(n) if q is None else np.asarray(q, dtype=float),
            np.zeros(n) if v is None else np.asarray(v, dtype=float),
        )
    )
    return program, resident.FixedRobotTree(program, initial, control_max_iterations=512)


def _action(program, *, target=None, rate=None, ff=None, effort=None, force=None, torque=None):
    n = program.model.dof_count

    def vector(value):
        return np.zeros(n) if value is None else np.asarray(value, dtype=float)

    def field(value):
        return np.zeros((n, 3)) if value is None else np.asarray(value, dtype=float)

    return program.inputs(
        vector(target), vector(rate), feedforward=vector(ff),
        loads=ph.RobotInputs(vector(effort), field(force), field(torque)),
    )


def _slider(*, origin=(0, 0, 3 / 32), q=0.0, v=0.0, kp=0.0, kd=0.0, delay=0):
    model = ph.RobotModel(
        np.array([-1]), ("prismatic",), np.array([origin], dtype=float),
        np.array([[0.0, 0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, 1.0]]),
        np.array([2.0]), np.zeros((1, 3)), np.full((1, 3), 1 / 8), np.zeros(3),
    )
    return _program(model, q=[q], v=[v], kp=[kp], kd=[kd], delay=delay)


def _r01_geometry(batch, world=17):
    moving, fixed = ph.CollisionKey(world, 100, 11), ph.CollisionKey(world, 900, 1)
    local = _box([-1 / 8] * 3, [1 / 8] * 3)
    support = _box([-1 / 2, -1 / 2, -1 / 4], [1 / 2, 1 / 2, 0])
    first = ph.CollisionShape(moving, "hull", tuple(range(101, 109)))
    second = ph.CollisionShape(fixed, "hull", tuple(range(201, 209)))
    model = ph.CollisionModel((first, second))
    topology = batch.snapshot([world])[world].topology
    bindings = (
        ph.ResidentTreeBinding(moving, world, 0, local, topology=topology),
        ph.ResidentStaticBinding(fixed, ph.CollisionMotion(support)),
    )
    return model, bindings, local, support


@pytest.mark.tyx
def test_r01_public_contact_step_follows_real_host_patch_two_newtons(tyx_mode, tmp_path):
    """Valid old geometry/law/tree first; RED is the missing public contact entry."""
    program, world = _slider()
    law = ph.RobotContactMaterial(stiffness=64, damping=0, friction=0)
    command = _action(program)
    with resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({17: command})
        model, bindings, local, support = _r01_geometry(batch)
        # This old HOST path is only a precondition, not the dynamics oracle.
        patch = contact_patch_sample(
            model.shapes[0], model.shapes[1], local + [0, 0, 3 / 32], support,
        )
        assert patch is not None and patch.dimension == 2
        assert abs(sum(site.weight for site in patch.sites) - 1) <= 1e-12
        sampled = sample_patch_loads(
            patch, law, np.zeros((len(patch.sites), 3)), np.zeros((len(patch.sites), 3)),
        )
        np.testing.assert_allclose(sampled["first_force"].sum(axis=0), [0, 0, 2], atol=1e-12, rtol=0)
        np.testing.assert_allclose(sampled["second_force"].sum(axis=0), [0, 0, -2], atol=1e-12, rtol=0)
        before = batch.snapshot()[17]
        _save(tmp_path, "r01-public-input", dict(
            h_s=H, model=program.model, command=command, collision_model=model,
            local_material=local, static_material=support, law=law,
            host_patch=patch, host_sampled=sampled, before=before, mode=tyx_mode,
        ))
        with batch.prepare_contacts(model, bindings, {(model.shapes[0].key, model.shapes[1].key): law}) as plan:
            assert batch.step_contacts(plan, H) == {17: True}
            after = batch.snapshot()[17]
            receipt = plan.last_accepted[17]
            assert plan.last_status[17] == "ACCEPTED"
            assert before.tick == 0 and after.tick == 1
            # All input values and this single response are exactly dyadic FP32.
            assert after.velocity == (H,) and after.position == (H * H,)
            assert after.applied_effort == (0.0,) and after.generalized_applied == (2.0,)
            assert after.control == before.control and after.history == before.history == ()
            assert (receipt.world, receipt.entry_tick, receipt.exit_tick, receipt.dt_s) == (17, 0, 1, H)
            assert receipt.entry_state == before and receipt.accepted_state == after
            _close(receipt.law_wrench_fp64, [[0, 0, 2, 0, 0, 0]])
            _close(receipt.uploaded_wrench_fp32, [[0, 0, 2, 0, 0, 0]])
            _close(receipt.contact_generalized, [2])
            _close(receipt.actual_drive, [0])
            _save(tmp_path, "r01-public-result", dict(
                before=before, after=after, receipt=receipt,
                diagnostic=batch.tree_diagnostics(), geometry=batch.tree_kinematics(), stats=batch.stats(),
            ))

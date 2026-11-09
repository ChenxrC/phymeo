"""Real TYX env/contact connections with analytic force and publication checks.

AUTHORING_ONLY / UNEXECUTED. No producer or HOST dynamics constructs the oracle.
All device execution belongs to the parent. Two-world overflow cases exercise
real FP32 rejection flags rather than mocking a physical kernel or acceptance.
"""

import gzip
import json
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

import phymeo as ph
from phymeo.core.rigid import RigidBodyModel
from phymeo.dynamics.floating_robot import FloatingRobotModel, FloatingRobotState
from phymeo.solvers.floating_tree import FloatingRobotInputs
from phymeo.training import ResidentContactStepper, ResidentVectorEnv, VectorEnvStepError

H = 1 / 128
WORLD = 17


def _encode(value):
    if isinstance(value, np.ndarray):
        return {"shape": value.shape, "dtype": value.dtype.str,
                "bytes_hex": value.tobytes().hex(), "values": _encode(value.tolist())}
    if isinstance(value, np.generic):
        return _encode(value.item())
    if is_dataclass(value):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_ieee_value": repr(value)}
    return value


def _save(path, name, value):
    with gzip.open(path / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(_encode(value), stream, allow_nan=False)


def _box(low, high):
    return np.array(list(product(*zip(low, high, strict=True))), dtype=float)


def _tree_case(kind):
    if kind == "fixed-box":
        model = ph.RobotModel(np.array([-1]), ("prismatic",),
                             np.array([[0., 0, 3 / 32]]), np.array([[0., 0, 0, 1]]),
                             np.array([[0., 0, 1]]), np.array([2.]), np.zeros((1, 3)),
                             np.full((1, 3), 1 / 8), np.zeros(3))
        program = ph.DrivenRobotProgram(model, kp=np.zeros(1), kd=np.zeros(1),
                                        effort_limit=np.array([8.]), delay_steps=0)
        state = program.initial_state(ph.RobotState(np.zeros(1), np.zeros(1)))
        tree = ph.FixedRobotTree(program, state)
        local = _box([-1 / 8] * 3, [1 / 8] * 3)
        return tree, model, program, local
    model = FloatingRobotModel(RigidBodyModel(np.array([.5]), np.full((1, 3), .5), np.zeros(3)),
                               np.zeros(3), None)
    state = FloatingRobotState(np.array([0., 0, 3 / 32, 0, 0, 0, 1]), np.zeros(6))
    tree = ph.FloatingRobotTree(model, state, kp=np.zeros(0), kd=np.zeros(0), effort_limit=np.zeros(0))
    return tree, model, None, np.array([[1 / 4, 0., 0.]])


def _geometry(batch, kind, local):
    moving, floor = ph.CollisionKey(WORLD, 100, 1), ph.CollisionKey(WORLD, 900, 2)
    fixed = _box([-1 / 2, -1 / 2, -1 / 4], [1 / 2, 1 / 2, 0])
    static_shape = ph.CollisionShape(floor, "hull", tuple(range(201, 209)))
    topology = batch.snapshot()[WORLD].topology
    if kind == "fixed-box":
        shape = ph.CollisionShape(moving, "hull", tuple(range(101, 109)))
        dynamic_binding = ph.ResidentTreeBinding(moving, WORLD, 0, local, topology=topology)
    else:
        shape = ph.CollisionShape(moving, "sphere", (101,), radius=1 / 8)
        dynamic_binding = ph.ResidentFloatingTreeBinding(moving, WORLD, 0, local, topology=topology)
    collision = ph.CollisionModel((shape, static_shape))
    bindings = (dynamic_binding, ph.ResidentStaticBinding(floor, ph.CollisionMotion(fixed)))
    law = ph.RobotContactMaterial(64, 0, 0)
    return collision, bindings, {(moving, floor): law}, fixed


class RobotTask:
    def __init__(self, kind, tree, program):
        self.kind, self.tree, self.program = kind, tree, program
        self.evaluate_calls = 0
        self.records = []

    def controls(self, actions, keys):
        if self.kind == "fixed-box":
            return {WORLD: self.program.inputs(np.zeros(1), np.zeros(1), feedforward=actions[0])}
        loads = FloatingRobotInputs(np.zeros(0), np.array([[0., 0, actions[0, 0]]]), np.zeros((1, 3)))
        return {WORLD: self.tree.inputs(np.zeros((0, 3)), external=loads)}

    def observe(self, states, keys):
        state = states[WORLD]
        q = state.position[0] if self.kind == "fixed-box" else state.position[2] - 3 / 32
        v = state.velocity[0] if self.kind == "fixed-box" else state.velocity[2]
        return np.array([[q, v]])

    def evaluate(self, before, after, actions, keys):
        self.evaluate_calls += 1
        self.records.append({"before": before, "after": after, "actions": actions})
        value = self.observe(after, keys)
        return value[:, 1], np.array([self.kind == "fixed-box"])


@pytest.mark.tyx
@pytest.mark.parametrize("kind", ("fixed-box", "floating-offset-sphere"))
def test_env_contact_stepper_advances_real_load_and_preserves_final_observation(kind, tyx_mode, tmp_path):
    tree, physical_model, program, local = _tree_case(kind)
    record = dict(kind=kind, mode=tyx_mode, h_s=H, physical_model=physical_model,
                  declared_local_geometry=local, expected_force_n=2., expected_acceleration_z=1.)
    try:
        with ph.TyxWorldBatch([(WORLD, tree)], tyx_mode, cache_root=tmp_path) as batch:
            collision, bindings, laws, fixed = _geometry(batch, kind, local)
            record.update(collision=collision, bindings=bindings, laws=laws, floor_points=fixed)
            with batch.prepare_contacts(collision, bindings, laws) as plan:
                stepper = ResidentContactStepper(batch, plan)
                task = RobotTask(kind, tree, program)
                env = ResidentVectorEnv(batch, task, (WORLD,), action_size=1, dt=H,
                                        horizon=1, stepper=stepper)
                initial = batch.snapshot()[WORLD]
                transition = env.step(np.zeros((1, 1)))
                receipt = plan.last_accepted[WORLD]
                accepted = task.records[0]["after"][WORLD]
                record.update(initial=initial, transition=transition, accepted=accepted,
                              receipt=receipt, task_records=task.records, after_reset=batch.snapshot(),
                              plan_stats=plan.stats(), batch_stats=batch.stats())
                # Depth=1/32, k=64 gives 2 N; mass=2 gives a_z=1, v+=h, q+=h^2.
                np.testing.assert_allclose(transition.final_observation, [[H * H, H]], rtol=0, atol=5e-6)
                np.testing.assert_allclose(transition.observation, [[0, 0]], rtol=0, atol=5e-6)
                np.testing.assert_allclose(transition.reward, [H], rtol=0, atol=5e-6)
                assert transition.final_mask.tolist() == [True]
                assert transition.terminated.tolist() == [kind == "fixed-box"]
                assert transition.truncated.tolist() == [kind != "fixed-box"]
                assert transition.episode_id.tolist() == [1] and transition.episode_length.tolist() == [0]
                assert accepted.tick == 1 and batch.snapshot()[WORLD] == initial
                assert task.evaluate_calls == 1 and not env.needs_reset
                if kind == "fixed-box":
                    np.testing.assert_allclose(receipt.law_wrench_fp64, [[0, 0, 2, 0, 0, 0]], rtol=0, atol=5e-6)
                    np.testing.assert_allclose(receipt.contact_generalized, [2], rtol=0, atol=5e-6)
                    np.testing.assert_allclose(receipt.diagnostics["mass"], [[2]], rtol=0, atol=5e-6)
                else:
                    # Eccentric x=1/4 site gives tau_y=-1/2, I_y=1/2 gives alpha_y=-1.
                    np.testing.assert_allclose(receipt.law_wrench_fp64, [[0, 0, 2, 0, -.5, 0]], rtol=0, atol=5e-6)
                    np.testing.assert_allclose(accepted.velocity, [0, 0, H, 0, -H, 0], rtol=0, atol=5e-6)
                    np.testing.assert_allclose(accepted.position[3:7],
                                               [0, -np.sin(H * H / 2), 0, np.cos(H * H / 2)],
                                               rtol=0, atol=5e-6)
                assert stepper.batch is batch and stepper.plan is plan
                record["status"] = "passed"
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        _save(tmp_path, "env-contact-" + kind, record)


class SphereTask:
    def __init__(self):
        self.evaluate_calls = 0

    def controls(self, actions, keys):
        return dict(zip(keys, actions, strict=True))

    def observe(self, states, keys):
        return np.array([(*states[key].position, *states[key].velocity) for key in keys])

    def evaluate(self, before, after, actions, keys):
        self.evaluate_calls += 1
        return np.zeros(len(keys)), np.zeros(len(keys), dtype=bool)


@pytest.mark.tyx
@pytest.mark.parametrize("all_rejected", (False, True), ids=("partial", "all-false"))
def test_real_two_world_rejection_mask_never_emits_a_training_transition(all_rejected, tyx_mode, tmp_path):
    keys = (4, 7)
    task = SphereTask()
    largest = float(np.finfo(np.float32).max)
    actions = np.array([[largest if all_rejected else 1., 0, 0], [largest, 0, 0]])
    record = dict(mode=tyx_mode, all_rejected=all_rejected, actions=actions, h_s=H,
                  physical_mass_kg=.5, cause="finite FP32 max force times inverse mass 2 overflows")
    try:
        with ph.TyxWorldBatch([(key, ph.FreeSphere(.5, .05, gravity=(0, 0, 0))) for key in keys],
                             tyx_mode, cache_root=tmp_path) as batch:
            env = ResidentVectorEnv(batch, task, keys, action_size=3, dt=H, horizon=5)
            before = batch.snapshot()
            with pytest.raises(VectorEnvStepError) as caught:
                env.step(actions)
            after = batch.snapshot()
            expected_mask = {4: not all_rejected, 7: False}
            record.update(before=before, after_attempt=after, mask=caught.value.accepted_worlds,
                          error=str(caught.value), physics_accepted=caught.value.physics_accepted,
                          acceptance_known=caught.value.acceptance_known, stats_after_attempt=batch.stats())
            assert dict(caught.value.accepted_worlds) == expected_mask
            assert caught.value.acceptance_known and caught.value.physics_accepted == (not all_rejected)
            assert env.needs_reset == (not all_rejected) and task.evaluate_calls == 0
            np.testing.assert_array_equal(env._length, [0, 0])
            np.testing.assert_array_equal(env._episode_id, [0, 0])
            assert after[7].tick == 0 and after[7].position == before[7].position and after[7].velocity == before[7].velocity
            if all_rejected:
                assert after[4].tick == 0 and after[4].position == before[4].position
            else:
                assert after[4].tick == 1
                np.testing.assert_allclose(after[4].velocity, [2 * H, 0, 0], rtol=0, atol=5e-6)
                np.testing.assert_allclose(after[4].position, [2 * H * H, 0, 0], rtol=0, atol=5e-6)
                with pytest.raises(RuntimeError, match="reset all"):
                    env.reset([0])
                with pytest.raises(RuntimeError, match="reset all"):
                    env.step(np.zeros((2, 3)))
                env.reset()
            transition = env.step(np.zeros((2, 3)))
            record.update(retry=transition, after_retry=batch.snapshot(), stats_after_retry=batch.stats())
            assert task.evaluate_calls == 1 and not env.needs_reset
            assert transition.episode_length.tolist() == [1, 1]
            assert transition.episode_id.tolist() == ([0, 0] if all_rejected else [1, 1])
            assert all(state.tick == 1 for state in batch.snapshot().values())
            record["status"] = "passed"
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        _save(tmp_path, "env-real-rejection-" + ("all" if all_rejected else "partial"), record)


@pytest.mark.tyx
def test_contact_stepper_checks_both_owner_edges_and_borrows_open_resources(tyx_mode, tmp_path):
    tree, _, _, local = _tree_case("fixed-box")
    with ph.TyxWorldBatch([(WORLD, tree)], tyx_mode, cache_root=tmp_path / "owner") as batch:
        collision, bindings, laws, _ = _geometry(batch, "fixed-box", local)
        with batch.prepare_contacts(collision, bindings, laws) as plan:
            stepper = ResidentContactStepper(batch, plan)
            with ph.TyxWorldBatch([(WORLD, ph.FreeSphere(1, .05))], tyx_mode,
                                 cache_root=tmp_path / "other") as other:
                with pytest.raises(ValueError, match="owner"):
                    ResidentContactStepper(other, plan)
                with pytest.raises(ValueError, match="owner"):
                    ResidentVectorEnv(other, SphereTask(), (WORLD,), action_size=3,
                                      dt=H, horizon=3, stepper=stepper)
                assert other.snapshot()[WORLD].tick == 0
            assert batch.snapshot()[WORLD].tick == 0 and plan.last_accepted == {}
            with pytest.raises(TypeError, match="ContactPlan"):
                ResidentContactStepper(batch, object())
            plan.close()
            with pytest.raises(RuntimeError, match="closed"):
                stepper(H)
            assert batch.snapshot()[WORLD].tick == 0

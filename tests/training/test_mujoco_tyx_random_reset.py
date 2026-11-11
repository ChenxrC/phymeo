"""N5 fixed sampler, actual CUDA episode resets and N3 continuation."""

import hashlib
import importlib.util
import random
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import pytest

KEYS = (7, 31, 103, 4096)
ROOT = Path(__file__).resolve().parents[2]


def example():
    spec = importlib.util.spec_from_file_location(
        "n5_random_reset_example", ROOT / "examples/mujoco_tyx_random_reset.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def plan(case="J", **kwargs):
    pytest.importorskip("mujoco")
    return example().components(case, keys=KEYS, **kwargs)["reset_plan"]


def sig(value):
    if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.tobytes()
    if is_dataclass(value):
        return type(value).__name__, tuple(
            (f.name, sig(getattr(value, f.name))) for f in fields(value)
        )
    if isinstance(value, Mapping):
        return tuple((key, sig(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(sig(item) for item in value)
    return value


def test_frozen_encoding_and_independent_hash_sampling():
    from phymeo.training.mujoco_tyx_reset import encoded_reset_key

    # Fixed bytes, not generated through the product encoder.
    golden = (
        b"phymeo-reset-v1\0\x02\x00N5"
        b"\x6f\x19\0\0\0\0\0\0"
        b"\x07\0\0\0"
        b"\0\0\0\0\0\0\0\0q\0\0\0\0"
    )
    assert encoded_reset_key("N5", 6511, 7, 0, "q", 0) == golden
    sampled = plan(namespace="N5").prepare((7,), dict.fromkeys(KEYS, 0))
    z = int.from_bytes(hashlib.sha256(golden).digest()[:8], "little") >> 11
    expected = np.float32(-0.1 + (z * 2.0 ** -53) * (0.1 - (-0.1)))
    assert sampled.requested[7].q[0] == float(expected)
    # Separate rebuild produces the same complete immutable sample.
    assert sig(sampled) == sig(plan(namespace="N5").prepare((7,), dict.fromkeys(KEYS, 0)))
    with pytest.raises(TypeError):
        sampled.next_indices[7] = 10
    with pytest.raises(ValueError):
        sampled.requested[7].q.setflags(write=True)


@pytest.mark.parametrize("case", ("J", "S"))
def test_sampler_subset_order_and_unused_streams(case):
    p = plan(case)
    counters = dict.fromkeys(KEYS, 0)
    union = p.prepare((103, 7), counters)
    reverse = p.prepare((7, 103), counters)
    assert sig(union) == sig(reverse)
    first = p.prepare((103,), counters)
    second = p.prepare((7,), first.next_indices)
    assert sig(first.requested[103]) == sig(union.requested[103])
    assert sig(second.requested[7]) == sig(union.requested[7])
    assert dict(second.next_indices) == dict(union.next_indices)
    assert union.next_indices[31] == union.next_indices[4096] == 0
    empty = p.prepare((), union.next_indices)
    assert dict(empty.next_indices) == dict(union.next_indices)
    assert empty.requested == {}
    from phymeo.training.mujoco_tyx_reset import MuJoCoTyxResetPlan
    parts = example().components(case)
    reordered = MuJoCoTyxResetPlan(
        parts["compilation"], dict(reversed(tuple(p.anchors.items()))),
        q_indices=p.q_indices, q_bounds=p.q_bounds,
        v_indices=p.v_indices, v_bounds=p.v_bounds, seed=p.seed, namespace=p.namespace,
    )
    other = reordered.prepare((103, 7), counters)
    assert reordered.sha256 != p.sha256  # Full order binding is deliberately distinct.
    for key in union.world_keys:
        assert sig(other.requested[key]) == sig(union.requested[key])
    before_numpy, before_python = np.random.get_state(), random.getstate()
    for index in range(3):
        p.prepare(KEYS, dict.fromkeys(KEYS, index))
    assert sig(np.random.get_state()) == sig(before_numpy)
    assert random.getstate() == before_python


@pytest.mark.parametrize("fault", ("duplicate", "unknown", "counter", "overflow", "bool-key"))
def test_bad_sample_does_not_consume_any_state(fault):
    p = plan()
    counters = dict.fromkeys(KEYS, 0)
    selected = (7,)
    if fault == "duplicate":
        selected = (7, 7)
    elif fault == "unknown":
        selected = (999,)
    elif fault == "bool-key":
        selected = (True,)
    elif fault == "counter":
        counters[31] = -1
    else:
        counters[7] = 2**64 - 1
    before = dict(counters)
    with pytest.raises((ValueError, OverflowError)):
        p.prepare(selected, counters)
    assert counters == before


@pytest.mark.parametrize("fault", ("q-address", "v-address", "shape", "nonfinite", "anchor", "time"))
def test_bad_plan_rejected_without_runtime(fault):
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState
    from phymeo.training.mujoco_tyx_reset import MuJoCoTyxResetPlan

    parts = example().components("J", keys=KEYS)
    args = dict(q_indices=(0, 1), q_bounds=((-0.1, 0.1),) * 2,
                v_indices=(0, 1), v_bounds=((-0.04, 0.04),) * 2,
                seed=6511, namespace="N5")
    anchors = dict(parts["anchors"])
    if fault == "q-address":
        args["q_indices"] = (0, 9)
    elif fault == "v-address":
        args["v_indices"] = (0, 9)
    elif fault == "shape":
        args["q_bounds"] = ((0, 1),)
    elif fault == "nonfinite":
        args["v_bounds"] = ((-np.inf, 1),) * 2
    elif fault == "anchor":
        anchors[7] = MuJoCoInitialState([0], [0])
    else:
        anchors[7] = MuJoCoInitialState([0, 0], [0, 0], time=1)
    with pytest.raises((ValueError, TypeError)):
        MuJoCoTyxResetPlan(parts["compilation"], anchors, **args)


def test_root_quaternion_and_velocity_are_not_randomizable():
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    from phymeo.training.mujoco_tyx_reset import MuJoCoTyxResetPlan

    compiled = compile_mujoco_xml(
        '<mujoco><worldbody><body><freejoint/><geom type="box" size=".1 .1 .1" '
        'mass="1" contype="0" conaffinity="0"/></body></worldbody></mujoco>',
        require_asset=False,
    )
    for channel in ("q", "v"):
        args = dict(q_indices=(), q_bounds=(), v_indices=(), v_bounds=(), seed=6511,
                    namespace="N5")
        args[channel + "_indices"] = (0,)
        args[channel + "_bounds"] = ((-0.1, 0.1),)
        with pytest.raises(ValueError, match="scalar|hinge|slide"):
            MuJoCoTyxResetPlan(compiled, {7: MuJoCoInitialState()}, **args)


@pytest.mark.cuda
@pytest.mark.tyx
@pytest.mark.parametrize("case", ("J", "S"))
def test_actual_manual_and_automatic_random_reset(case, tmp_path):
    api = example()
    torch = api.require_cuda()
    with api.make_environment(case) as env:
        original = env.get_observations()
        state = env.runtime.snapshot()
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(env.device).clone()
        env.reset((103, 7))
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(env.device), cuda_rng)
        assert dict(env.reset_indices) == {7: 1, 31: 0, 103: 1, 4096: 0}
        for key in (31, 4096):
            assert sig(env.runtime.snapshot()[key]) == sig(state[key])
        assert not torch.equal(original, env.get_observations())
        transitions = []
        for _ in range(6):
            action = torch.zeros((4, env.num_actions), device=env.device)
            tr = env.step(action, world_keys=KEYS, expected_frame=env.current_frame)
            assert all(tr.accepted_worlds.values())
            transitions.append(tr)
            for row, key in enumerate(KEYS):
                raw = np.r_[tr.states[key].q, tr.states[key].v].astype(np.float32)
                expected = raw * env.task_spec.scale + env.task_spec.bias
                np.testing.assert_allclose(tr.final_observation[row].cpu(), expected,
                                           rtol=2e-5, atol=2e-6)
                if bool(tr.final_mask[row]):
                    assert env.runtime.snapshot()[key].tick == 0
                    assert not torch.equal(tr.observation[row], tr.final_observation[row])
            if case == "S":
                assert all(any(c.normal_force > 0 for c in s.last_step.contacts)
                           for s in tr.states.values())
        assert any(bool(t.final_mask.any()) and not bool(t.final_mask.all()) for t in transitions)
        assert env.reset_indices[7] > 1
        (tmp_path / (case + ".json")).write_text(
            api.json_text({"binding": dict(env.binding), "transitions": transitions,
                           "random_reset": env.capture_training_state()["random_reset"]}),
            encoding="utf-8",
        )


@pytest.mark.cuda
@pytest.mark.tyx
@pytest.mark.parametrize("phase", ("before-cas", "after-cas"))
def test_failed_reset_rng_ledger_matches_real_commit(phase, monkeypatch):
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxEnvError

    with example().make_environment("J") as env:
        before = env.runtime.checkpoint().to_bytes()
        indices = dict(env.reset_indices)
        if phase == "before-cas":
            original = env.runtime.reset_states
            def fail(*args, **kwargs):
                raise RuntimeError("injected pre-CAS reset failure")
            monkeypatch.setattr(env.runtime, "reset_states", fail)
        else:
            original = env._export
            def fail():
                raise RuntimeError("injected post-CAS export failure")
            monkeypatch.setattr(env, "_export", fail)
        with pytest.raises(MuJoCoTyxEnvError) as caught:
            env.reset((7,))
        if phase == "before-cas":
            assert not caught.value.reset_accepted
            assert env.runtime.checkpoint().to_bytes() == before
            assert dict(env.reset_indices) == indices
            assert not env.needs_recovery
            monkeypatch.setattr(env.runtime, "reset_states", original)
            env.reset((7,))
            assert env.reset_indices[7] == 1
        else:
            assert caught.value.reset_accepted and env.needs_recovery
            assert env.runtime.checkpoint().to_bytes() != before
            assert env.reset_indices[7] == 1 and env.last_reset_receipt is not None
            with pytest.raises(RuntimeError, match="recovery|recover"):
                env.reset((31,))
            monkeypatch.setattr(env, "_export", original)
            env.reset()
            assert env.reset_indices[7] == 2
            assert all(env.reset_indices[key] == 1 for key in KEYS[1:])
            assert not env.needs_recovery


@pytest.mark.cuda
@pytest.mark.tyx
def test_external_mutation_requires_all_world_recovery():
    with example().make_environment("J") as env:
        env.runtime.reset((7,))
        with pytest.raises(RuntimeError, match="recovery"):
            env.reset((31,))
        assert dict(env.reset_indices) == dict.fromkeys(KEYS, 0)
        env.reset()
        assert dict(env.reset_indices) == dict.fromkeys(KEYS, 1)
        assert not env.needs_recovery


@pytest.mark.cuda
@pytest.mark.tyx
@pytest.mark.parametrize("case,recurrent", (("J", False), ("S", False), ("J", True)))
def test_n3_restores_next_random_reset_and_real_update(case, recurrent, tmp_path):
    result = example().continuation(case, tmp_path, recurrent=recurrent)
    assert result["status"] == "PASS_RANDOM_RESET_CONTINUATION"
    assert result["transitions"] == 32
    assert result["reset_count_advanced"]
    if recurrent:
        assert result["nonzero_hidden"] and result["partial_done_masks"]


@pytest.mark.cuda
@pytest.mark.tyx
def test_saved_counter_bad_binding_and_late_restore(monkeypatch):
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxEnvError

    with example().make_environment("J") as env:
        env.reset()
        saved = env.capture_training_state()
        capsule = env.runtime.checkpoint()
        for field, value in (("plan_sha256", "0" * 64), ("next_indices", {7: -1})):
            bad = dict(saved)
            bad["random_reset"] = {**saved["random_reset"], field: value}
            before = env.runtime.checkpoint().to_bytes()
            with pytest.raises(ValueError):
                env.restore_training_state(
                    capsule, bad, expected_revision=env.runtime.checkpoint_revision
                )
            assert env.runtime.checkpoint().to_bytes() == before
        env.reset()
        original = env._export
        def fail():
            raise RuntimeError("injected restored physics export failure")
        monkeypatch.setattr(env, "_export", fail)
        with pytest.raises(MuJoCoTyxEnvError) as caught:
            env.restore_training_state(
                capsule, saved, expected_revision=env.runtime.checkpoint_revision
            )
        assert caught.value.state_accepted and env.needs_recovery
        assert dict(env.reset_indices) == dict.fromkeys(KEYS, 1)
        monkeypatch.setattr(env, "_export", original)


@pytest.mark.cuda
@pytest.mark.tyx
def test_clone_composition_preserves_next_sample():
    with example().make_environment("J") as env:
        env.reset((7, 103))
        state = env.capture_training_state()
        capsule = env.runtime.checkpoint()
        with env.runtime.clone() as runtime:
            with example().make_environment("J", runtime=runtime) as other:
                other.restore_training_state(
                    capsule, state, expected_revision=runtime.checkpoint_revision
                )
                env.reset((31, 7))
                other.reset((7, 31))
                assert sig(env.runtime.snapshot()) == sig(other.runtime.snapshot())
                assert sig(env.capture_training_state()) == sig(other.capture_training_state())


def test_seed_namespace_and_saved_counter_identity_are_bound():
    original = plan()
    sample = original.prepare((7,), dict.fromkeys(KEYS, 0))
    saved = {"plan_sha256": original.sha256, "next_indices": dict(sample.next_indices),
             "last_sample": {"world_keys": sample.world_keys, "indices": dict(sample.indices)}}
    original.validate_training_state(saved)
    for other in (plan(seed=6512), plan(namespace="N5-other")):
        assert other.sha256 != original.sha256
        with pytest.raises(ValueError, match="binding"):
            other.validate_training_state(saved)
    bad = {**saved, "last_sample": {"world_keys": (7,), "indices": {7: 5}}}
    with pytest.raises(ValueError, match="counter"):
        original.validate_training_state(bad)
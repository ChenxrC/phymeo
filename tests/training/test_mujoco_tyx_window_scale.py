"""Fixed N7 protocol, complete comparison and bounded failure evidence."""

import importlib.util
from dataclasses import make_dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def api():
    path = Path(__file__).resolve().parents[2] / "examples/mujoco_tyx_window_scale.py"
    spec = importlib.util.spec_from_file_location("n7_window_scale_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "task,worlds,d,recurrent,total",
    (("J", 8, 2, True, 384), ("S", 128, 4, False, 12288), ("S", 256, 4, False, 24576)),
)
def test_fixed_protocol_and_actual_counts(task, worlds, d, recurrent, total):
    module = api()
    row = module.protocol(task, worlds)
    assert row["decimation"] == d and row["recurrent"] is recurrent
    assert row["keys"] == tuple(7 + 13 * i for i in range(worlds))
    assert row["horizons"] == tuple((3, 5, 7, 11)[i % 4] for i in range(worlds))
    assert row["updates"] == 3 and row["rollout_steps"] == 8
    assert row["physical_worldsteps"] == total
    assert row["measured_worldsteps"] == 16 * d * worlds
    assert row["seed"] == 6511 and row["namespace"] == "N7"


@pytest.mark.parametrize("task,worlds", (("J", 128), ("S", 8), ("R", 8), ("J", True), ("S", 257)))
def test_no_extra_task_or_capacity(task, worlds):
    with pytest.raises(ValueError):
        api().protocol(task, worlds)


def frame(owner):
    module = api().support()
    binding = {key: key for key in module.TASK_BINDING_FIELDS}
    binding.update(runtime_owner=owner, workspace_owner=owner)
    return SimpleNamespace(binding=binding, revision=7, epoch=2, ticks=(2,), times=(0.004,))


def transition(owner="first", full=True):
    module = api()
    state = SimpleNamespace(**{name: (name, 1) for name in module.support().STATE_FIELDS})
    state.body_positions = "full-only" if full else None
    receipt = SimpleNamespace(
        binding={"runtime_owner": owner, "model_sha256": "m", "epoch": 2},
        source_revision=6,
        revision=7,
        epoch=2,
        controls={7: "held"},
    )
    StateReceipt = make_dataclass(
        "ActionReceipt", [(name, object) for name in vars(receipt)], frozen=True
    )
    receipt = StateReceipt(**vars(receipt))
    Sub = make_dataclass(
        "Sub",
        [
            (name, object)
            for name in (
                "ordinal",
                "entry_revision",
                "exit_revision",
                "accepted_worlds",
                "states",
                "diagnostics_complete",
                "error",
            )
        ],
        frozen=True,
    )
    sub = Sub(0, 6, 7, {7: True}, {7: state}, True, None)
    Window = make_dataclass(
        "Window",
        [
            (name, object)
            for name in (
                "spec_sha256",
                "physics_steps",
                "attempted",
                "completed",
                "substeps",
                "action_receipt",
                "phase",
                "failed_substep",
            )
        ],
        frozen=True,
    )
    window = Window("spec", 1, 1, 1, (sub,), receipt, "complete", None)
    value = {
        "input_frame": frame(owner),
        "output_frame": frame(owner),
        "action_receipt": receipt,
        "states": {7: state},
        "control_window": window,
        "action": np.array([[0.2]], np.float32),
        "future_common": ("retained", 1),
    }
    Tr = make_dataclass("Transition", [(name, object) for name in value], frozen=True)
    return Tr(**value)


def test_comparison_only_removes_opaque_owner_and_full_only_fields():
    module = api()
    left, right = transition(), transition("second", False)
    assert module.signature(module.common_transition(left)) == module.signature(
        module.common_transition(right)
    )
    right.input_frame.binding["model_sha256"] = "wrong"
    assert module.signature(module.common_transition(left)) != module.signature(
        module.common_transition(right)
    )


@pytest.mark.parametrize(
    "field",
    (
        "q",
        "v",
        "last_step",
        "held_control",
        "future_common",
        "revision",
        "substep_revision",
        "action",
    ),
)
def test_comparison_keeps_common_force_motion_and_publication(field):
    module = api()
    left, right = transition(), transition("second", False)
    if field in ("q", "v", "last_step", "held_control"):
        setattr(right.states[7], field, "different")
    elif field == "revision":
        right.input_frame.revision += 1
    elif field == "substep_revision":
        object.__setattr__(right.control_window.substeps[0], "exit_revision", 8)
    elif field == "action":
        right.action[0, 0] += 0.1
    else:
        object.__setattr__(right, field, "different")
    assert module.signature(module.common_transition(left)) != module.signature(
        module.common_transition(right)
    )


def test_public_observer_keeps_prefix_and_failed_window_and_restores_method():
    module = api()
    failure = RuntimeError("second public step failed")
    failure.control_window = "accepted-one-substep"
    failure.action_accepted = True

    class Adapter:
        calls = 0

        def step(self, action):
            self.calls += 1
            if self.calls == 2:
                raise failure
            self.last_transition = "accepted"
            return (np.array([1.0]), np.array([2.0]), np.array([False]), {"field": 3})

    env = SimpleNamespace(last_reset_receipt="reset", capture_training_state=lambda: {"index": 1})
    adapter = Adapter()
    original = adapter.step
    with module.observe_windows(adapter, env) as attempts:
        adapter.step(np.array([0.1]))
        with pytest.raises(RuntimeError, match="second"):
            adapter.step(np.array([0.2]))
    assert adapter.step == original and adapter.calls == 2
    assert attempts[0]["transition"] == "accepted" and attempts[0]["status"] == "accepted"
    assert attempts[1]["control_window"] == "accepted-one-substep"
    assert attempts[1]["action_accepted"] is True
    assert attempts[0]["reset"] == "reset"


def test_owning_observer_copies_rsl_outputs():
    module = api()
    value = {"a": np.array([0.25]), "b": (np.array([2]),)}
    owned = module.owned(value)
    value["a"][0] = 99
    assert owned["a"][0] == 0.25
    assert not np.shares_memory(value["b"][0], owned["b"][0])


def test_post_acceptance_evidence_failure_keeps_actual_transition():
    module = api()

    class Adapter:
        calls = 0

        def step(self, action):
            self.calls += 1
            self.last_transition = "real-accepted-publication"
            return (np.array([1.0]), np.array([2.0]), np.array([False]), {})

    def failed_capture():
        raise RuntimeError("post-acceptance evidence failure")

    adapter = Adapter()
    env = SimpleNamespace(last_reset_receipt=None, capture_training_state=failed_capture)
    with module.observe_windows(adapter, env) as attempts:
        with pytest.raises(RuntimeError, match="post-acceptance"):
            adapter.step(np.array([0.1]))
    assert adapter.calls == 1
    assert attempts[0]["transition"] == "real-accepted-publication"
    assert attempts[0]["status"] == "failed"
    assert attempts[0]["rsl_output"][0][0] == 1.0


def test_mode_boundary_rejects_live_owners():
    with pytest.raises(TypeError):
        api().mode_summary({"owner": object()})


@pytest.mark.cuda
@pytest.mark.parametrize("task,worlds", (("J", 8), ("S", 128), ("S", 256)))
def test_actual_fixed_paired_window_scale(task, worlds, tmp_path):
    result = api().run_pair(task, worlds, tmp_path / "result")
    assert result["status"] == "PASS_N7_PAIRED_WINDOW_SCALE"
    assert result["common_physics_training_identical"]
    for mode in ("full", "integration"):
        row = result["modes"][mode]
        p = result["protocol"]
        assert row["physical_worldsteps"] == p["physical_worldsteps"]
        assert row["counter_end"]["transitions"] == 24
        assert row["counter_end"]["physical_steps"] == 24 * p["decimation"]
        assert row["counter_end"]["resets"] > 0
        assert row["loop_to_trace_close_s"] >= row["compute_and_mandatory_diagnostics_s"] > 0
        assert len(row["signatures"]) == 3
        assert row["native_observation"]["constructor_failures"] == 0
        assert row["native_observation"]["copy_failures"] == 0
        if task == "J":
            assert row["partial_gru_masks"] > 0 and row["nonzero_gru_hidden"]

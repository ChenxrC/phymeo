"""Fixed N3 same-runtime continuation evidence using the accepted N2 examples.

Run only in the qualified CUDA training profile. Full physics HOST diagnostics
remain active. This is an update-boundary reproduction check, not a PPO benchmark.
"""

import argparse
import importlib.util
import json
import random
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np


def n2():
    path = Path(__file__).with_name("train_mujoco_tyx_ppo.py")
    spec = importlib.util.spec_from_file_location("phymeo_n3_accepted_n2_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact(left, right, path="state"):
    import torch

    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert left.detach().cpu().numpy().tobytes() == right.detach().cpu().numpy().tobytes(), path
    elif isinstance(left, np.ndarray):
        assert (
            isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
        ), path
        assert left.tobytes() == right.tobytes(), path
    elif is_dataclass(left):
        assert type(left) is type(right), path
        for field in fields(left):
            exact(getattr(left, field.name), getattr(right, field.name), path + "." + field.name)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping) and set(left) == set(right), path
        for key in left:
            exact(left[key], right[key], path + "." + str(key))
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, value in enumerate(left):
            exact(value, right[index], path + "[" + str(index) + "]")
    else:
        assert type(left) is type(right) and left == right, (path, left, right)


@contextmanager
def make_session(case):
    from phymeo.training.mujoco_tyx_checkpoint import MuJoCoTyxPpoSession
    from phymeo.training.rsl_mujoco_tyx import RslMuJoCoTyxVecEnv

    if case not in ("hinge-slide", "support-slide", "gru"):
        raise ValueError("unknown frozen N3 case")
    previous = n2()
    torch = previous.require_cuda()
    seed = {"hinge-slide": 11011, "support-slide": 11012, "gru": 11013}[case]
    previous._setup(torch, seed)
    random.seed(seed)
    with previous.make_environment("hinge-slide" if case == "gru" else case) as env:
        generators = {
            "torch-cuda": torch.Generator(device=env.device).manual_seed(seed + 1),
            "torch-cpu": torch.Generator(device="cpu").manual_seed(seed + 2),
            "numpy": np.random.default_rng(seed + 3),
            "python": random.Random(seed + 4),
        }
        session = MuJoCoTyxPpoSession(
            RslMuJoCoTyxVecEnv(env),
            steps=8 if case == "gru" else 16,
            recurrent=case == "gru",
            generators=generators,
        )
        # Example-owned references, registered explicitly rather than discovered.
        session.example_generators = generators
        yield session


def _draw(session):
    import torch

    named = session.example_generators
    return {
        "global_cpu": torch.rand(5),
        "global_cuda": torch.rand(7, device=session.env.device),
        "global_numpy": np.random.random(3),
        "global_python": tuple(random.random() for _ in range(3)),
        "named_cpu": torch.rand(5, generator=named["torch-cpu"]),
        "named_cuda": torch.rand(7, device=session.env.device, generator=named["torch-cuda"]),
        "named_numpy": named["numpy"].random(3),
        "named_python": tuple(named["python"].random() for _ in range(3)),
    }


def _comparable_transition(value):
    """Exclude only monotonic destination publication identities, never physics."""
    result = {field.name: getattr(value, field.name) for field in fields(value)}
    for name in ("input_frame", "output_frame"):
        frame = result[name]
        result[name] = {"binding": frame.binding, "ticks": frame.ticks, "times": frame.times}
    receipt = value.action_receipt
    result["action_receipt"] = {
        "binding": {key: item for key, item in receipt.binding.items() if key != "epoch"},
        "controls": receipt.controls,
    }
    return result


def _check_masks(record):
    import torch

    partial = 0
    for entry in record["recurrent_masks"]:
        done = entry["done"]
        if bool(done.any().item()) and not bool(done.all().item()):
            partial += 1
        for owner in ("actor", "critic"):
            before, after = entry["before"][owner], entry["after"][owner]
            exact(before[:, ~done, :], after[:, ~done, :], owner + ".non-done-hidden")
            if bool(done.any().item()):
                assert torch.count_nonzero(before[:, done, :]).item(), owner
                assert not torch.count_nonzero(after[:, done, :]).item(), owner
    return partial


def compare_case(case, output):
    previous = n2()
    n1 = previous.load_n1()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / (case + ".jsonl")
    with log_path.open("x", encoding="utf-8") as log:
        try:
            with make_session(case) as session:
                n1.append(
                    log,
                    {
                        "event": "configuration",
                        "case": case,
                        "binding": dict(session.env.binding),
                        "steps": session.steps,
                        "scope": "same-runtime-update-boundary-exact-continuation",
                        "monotonic_exclusions": [
                            "task-frame.revision",
                            "task-frame.epoch",
                            "action-receipt.source_revision",
                            "action-receipt.revision",
                            "action-receipt.epoch",
                            "action-receipt.binding.epoch",
                            "operational-counters",
                        ],
                    },
                )
                partial = 0
                for _ in range(1 if case == "gru" else 2):
                    result = session.update()
                    partial += _check_masks(result)
                    n1.append(log, {"event": "warm-update", "record": previous.diagnostic(result)})
                before = session.training_state()
                nonzero = {}
                if case == "gru":
                    import torch

                    nonzero = {
                        name: bool(torch.count_nonzero(value).item())
                        for name, value in before["hidden"].items()
                    }
                    assert all(nonzero.values())
                saved_states = session.env.runtime.snapshot()
                save = session.save_checkpoint(output / (case + "-bundle"))
                n1.append(
                    log,
                    {
                        "event": "checkpoint",
                        "save": save,
                        "training": previous.diagnostic(before),
                        "physics": saved_states,
                    },
                )
                random_a = _draw(session)
                record_a = session.update()
                partial += _check_masks(record_a)
                training_a = session.training_state()
                physical_a = session.env.runtime.snapshot()
                old_frame = session.env.current_frame
                revision_a = session.env.runtime.checkpoint_revision
                counters_a = dict(session.env.runtime.counters)
                n1.append(
                    log,
                    {
                        "event": "continuation-A",
                        "random": previous.diagnostic(random_a),
                        "update": previous.diagnostic(record_a),
                        "training": previous.diagnostic(training_a),
                        "physical": physical_a,
                        "counters": counters_a,
                    },
                )
                restored = session.restore_checkpoint(
                    output / (case + "-bundle"), expected_revision=revision_a
                )
                revision_restored = session.env.runtime.checkpoint_revision
                assert revision_restored > revision_a
                counters_restored = dict(session.env.runtime.counters)
                for name in (
                    "completed_steps",
                    "accepted_world_steps",
                    "failed_batches",
                    "device_control_submissions",
                    "device_control_refusals",
                ):
                    assert counters_restored[name] >= counters_a[name], name
                exact(saved_states, session.env.runtime.snapshot(), "restored-physics")
                exact(before, session.training_state(), "restored-training")
                try:
                    session.env.workspace.to_torch(old_frame)
                except (ValueError, RuntimeError):
                    stale_refused = True
                else:
                    raise AssertionError("old task frame survived physical restore")
                n1.append(
                    log,
                    {
                        "event": "restored",
                        "receipt": restored,
                        "training": previous.diagnostic(session.training_state()),
                        "physical": session.env.runtime.snapshot(),
                        "old_frame_refused": stale_refused,
                    },
                )
                random_b = _draw(session)
                record_b = session.update()
                partial += _check_masks(record_b)
                training_b = session.training_state()
                physical_b = session.env.runtime.snapshot()
                n1.append(
                    log,
                    {
                        "event": "continuation-B",
                        "random": previous.diagnostic(random_b),
                        "update": previous.diagnostic(record_b),
                        "training": previous.diagnostic(training_b),
                        "physical": physical_b,
                        "counters": dict(session.env.runtime.counters),
                    },
                )
                exact(random_a, random_b, "RNG-consumption")
                exact(training_a, training_b, "training-continuation")
                exact(physical_a, physical_b, "physical-continuation")
                assert len(record_a["transitions"]) == len(record_b["transitions"]) == session.steps
                for index, (left, right) in enumerate(
                    zip(record_a["transitions"], record_b["transitions"], strict=True)
                ):
                    assert all(left.accepted_worlds.values()) and all(
                        right.accepted_worlds.values()
                    )
                    assert right.input_frame.revision > left.input_frame.revision
                    assert right.input_frame.epoch > left.input_frame.epoch
                    assert right.action_receipt.revision > left.action_receipt.revision
                    assert right.action_receipt.epoch > left.action_receipt.epoch
                    exact(
                        _comparable_transition(left),
                        _comparable_transition(right),
                        "transition[" + str(index) + "]",
                    )
                for name in (
                    "iteration",
                    "next_iteration",
                    "loss",
                    "optimization",
                    "recurrent_masks",
                ):
                    exact(record_a[name], record_b[name], name)
                summary = {
                    "status": "PASS_N3_EXACT_CONTINUATION",
                    "case": case,
                    "next_iteration": session.next_iteration,
                    "compared_world_steps": session.steps * session.env.num_envs,
                    "physics_and_training_exact": True,
                    "old_frame_refused": stale_refused,
                    "destination_revision_increased": revision_restored > revision_a,
                    "saved_nonzero_actor_hidden": nonzero.get("actor", False),
                    "saved_nonzero_critic_hidden": nonzero.get("critic", False),
                    "partial_done_steps": partial,
                    "training_quality": "not-evaluated",
                    "full_GPU_training": False,
                    "module_origins": n1.module_origins(),
                }
                n1.append(log, {"event": "summary", **summary})
                return summary
        except BaseException as error:
            n1.append(
                log,
                {
                    "event": "failure",
                    "case": case,
                    "type": type(error).__name__,
                    "message": str(error),
                },
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("hinge-slide", "support-slide", "gru"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_case(args.case, args.output), sort_keys=True))


if __name__ == "__main__":
    main()

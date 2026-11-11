"""Public N5 scalar random resets and short exact training continuation.

Uses the unchanged N4 J/S model declarations. Sampling is HOST; task/action and
PPO are CUDA. The optional example is not a learning or performance benchmark.
"""

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

KEYS = (7, 31, 103, 4096)
HORIZONS = (3, 5, 7, 11)


def tasks():
    path = Path(__file__).with_name("mujoco_tyx_learning_tasks.py")
    spec = importlib.util.spec_from_file_location("n5_original_learning_tasks", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_cuda():
    return tasks().require_cuda()


def components(case, *, keys=KEYS, seed=6511, namespace="N5"):
    """Public case factory; no physical runtime or observation is created here."""
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState
    from phymeo.backends.mujoco_task_workspace import MuJoCoTyxTaskSpec
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxActionSpec
    from phymeo.training.mujoco_tyx_reset import MuJoCoTyxResetPlan

    if case not in ("J", "S") or tuple(keys) != KEYS:
        raise ValueError("N5 example fixes J/S and four sparse world keys")
    original = tasks()
    item = original.task(case)
    compiled = compile_mujoco_xml(item.xml, require_asset=False)
    width = len(item.q_target)
    initial = original.templates(case, held_out=False)
    anchors = {
        key: MuJoCoInitialState(*initial[index]) for index, key in enumerate(keys)
    }
    q_bounds = ((-0.1, 0.1),) * width if case == "J" else ((0.00065, 0.00085),)
    v_bounds = ((-0.04, 0.04),) * width if case == "J" else ((-0.002, 0.002),)
    reset_plan = MuJoCoTyxResetPlan(
        compiled, anchors, q_indices=tuple(range(width)), q_bounds=q_bounds,
        v_indices=tuple(range(width)), v_bounds=v_bounds, seed=seed, namespace=namespace,
    )
    spec = MuJoCoTyxTaskSpec(
        tuple(range(width)), tuple(range(width)),
        scale=tuple(1 / x for x in item.q_scale) + tuple(1 / x for x in item.v_scale),
        bias=tuple(-q / x for q, x in zip(item.q_target, item.q_scale, strict=True))
             + (0.0,) * width,
        target=item.q_target + (0.0,) * width,
        weight=tuple(1 / (width * x * x) for x in item.q_scale)
               + tuple(0.01 / (width * x * x) for x in item.v_scale),
        lower=item.q_lower + (-item.v_bound,) * width,
        upper=item.q_upper + (item.v_bound,) * width,
    )
    action = MuJoCoTyxActionSpec(
        tuple(range(width)), scale=[item.action_scale] * width, bias=[0.0] * width
    )
    return {"compilation": compiled, "anchors": anchors, "reset_plan": reset_plan,
            "task_spec": spec, "action_spec": action, "h": item.h}


@contextmanager
def make_environment(case, *, runtime=None, seed=6511, namespace="N5", horizons=HORIZONS):
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxVectorEnv

    require_cuda()
    parts = components(case, seed=seed, namespace=namespace)
    with ExitStack() as owners:
        if runtime is None:
            runtime = owners.enter_context(MuJoCoRuntime(
                parts["compilation"], mode="cuda", world_keys=KEYS,
                initial_states=parts["anchors"],
            ))
        env = owners.enter_context(MuJoCoTyxVectorEnv(
            runtime, parts["task_spec"], parts["action_spec"],
            h=parts["h"], horizons=horizons, reset_plan=parts["reset_plan"],
        ))
        yield env


def _encode(value):
    if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape),
                "data_hex": value.tobytes().hex()}
    if is_dataclass(value):
        return {"record": type(value).__name__,
                "fields": {f.name: _encode(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_text(value):
    return json.dumps(_encode(value), allow_nan=False, sort_keys=True, separators=(",", ":"))


def _comparable_transition(value):
    # Preserve all dynamics/task/reset data; only publication identities are
    # monotonic at the destination and cannot rewind with a physics capsule.
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


def _mask_checks(record):
    import torch

    partial = 0
    for entry in record["recurrent_masks"]:
        done = entry["done"]
        if bool(done.any()) and not bool(done.all()):
            partial += 1
        for name in ("actor", "critic"):
            before, after = entry["before"][name], entry["after"][name]
            if not torch.equal(before[:, ~done], after[:, ~done]):
                raise AssertionError("non-done recurrent hidden changed during mask")
            if bool(done.any()) and bool(torch.count_nonzero(after[:, done])):
                raise AssertionError("done recurrent hidden was not cleared")
    return partial


def continuation(case, output, *, recurrent=False):
    """One real warm update, save, one update A, restore, identical update B."""
    from phymeo.training.mujoco_tyx_checkpoint import MuJoCoTyxPpoSession
    from phymeo.training.rsl_mujoco_tyx import RslMuJoCoTyxVecEnv

    torch = require_cuda()
    tasks().seed_all(6511)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = case + ("-gru" if recurrent else "-mlp")
    result = {"status": "FAILED_RANDOM_RESET_CONTINUATION", "case": case,
              "recurrent": recurrent, "seed": 6511, "scope": "same-profile-exact-continuation"}
    with (output / (name + ".jsonl")).open("x", encoding="utf-8") as log:
        with make_environment(case) as env:
            try:
                env.reset()
                session = MuJoCoTyxPpoSession(
                    RslMuJoCoTyxVecEnv(env), steps=8, recurrent=recurrent
                )
                warm = session.update()
                log.write(json_text({"event": "warm-update", "record": warm,
                                     "random_reset": env.capture_training_state()["random_reset"]}) + "\n")
                before = session.training_state()
                nonzero = recurrent and all(
                    bool(torch.count_nonzero(value)) for value in before["hidden"].values()
                )
                if recurrent and not nonzero:
                    raise AssertionError("real nonzero actor/critic hidden required")
                checkpoint = output / (name + "-bundle")
                saved = session.save_checkpoint(checkpoint)
                indices = dict(env.reset_indices)
                record_a = session.update()
                final_a = session.training_state()
                physical_a = env.runtime.snapshot()
                log.write(json_text({"event": "A", "record": record_a,
                                     "training": final_a, "physics": physical_a}) + "\n")
                restored = session.restore_checkpoint(
                    checkpoint, expected_revision=env.runtime.checkpoint_revision
                )
                record_b = session.update()
                final_b = session.training_state()
                physical_b = env.runtime.snapshot()
                log.write(json_text({"event": "B", "record": record_b,
                                     "training": final_b, "physics": physical_b}) + "\n")
                if json_text(final_a) != json_text(final_b):
                    raise AssertionError("complete training/Adam/RNG/reset state differs")
                if json_text(physical_a) != json_text(physical_b):
                    raise AssertionError("complete physical snapshot differs")
                if json_text(tuple(map(_comparable_transition, record_a["transitions"]))) != json_text(
                    tuple(map(_comparable_transition, record_b["transitions"]))
                ):
                    raise AssertionError("complete continuation transitions differ")
                if json_text(record_a["loss"]) != json_text(record_b["loss"]):
                    raise AssertionError("upstream losses differ")
                advanced = any(env.reset_indices[key] > indices[key] for key in KEYS)
                if not advanced:
                    raise AssertionError("continuation did not cross random reset")
                masks = _mask_checks(warm) + _mask_checks(record_a) + _mask_checks(record_b)
                result.update(
                    status="PASS_RANDOM_RESET_CONTINUATION",
                    transitions=len(record_b["transitions"]) * len(KEYS),
                    reset_count_advanced=advanced, nonzero_hidden=bool(nonzero),
                    partial_done_masks=masks, save=saved, restore=restored,
                    reset_indices=dict(env.reset_indices),
                    binding=env.training_binding(),
                    exclusions=["destination-CAS-revision", "task-epoch", "resource-counters",
                                "action-receipt-publication-revision/epoch"],
                )
            except BaseException as error:
                log.write(json_text({"event": "failure", "error": repr(error),
                                     "needs_recovery": env.needs_recovery,
                                     "reset_indices": dict(env.reset_indices)}) + "\n")
                raise
            finally:
                (output / (name + "-summary.json")).write_text(json_text(result), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("J", "S"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recurrent", action="store_true")
    args = parser.parse_args(argv)
    if args.recurrent and args.case != "J":
        parser.error("the fixed recurrent diagnostic uses J")
    result = continuation(args.case, args.output, recurrent=args.recurrent)
    print(json_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

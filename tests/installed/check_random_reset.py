"""Installed N5 public random-reset consumer; copy outside the source checkout.

Only CUDA is valid for MuJoCoTyxVectorEnv. The inline XML reproduces the original
N4 J task (_joint_xml), with no physics/model edits. No source examples or tests
are imported. CPU MuJoCo compilation/detached observation remain explicit;
CPU integration is forbidden. Eight short batch steps, no PPO/training run.
"""

import argparse
import base64
import gzip
import hashlib
import json
import struct
import sys
import traceback
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

H = 0.002
KEYS = (7, 31, 103, 4096)
XML = (
    '<mujoco model="n4-j"><compiler angle="radian" fusestatic="false"/>'
    '<option timestep=".002" gravity="0 0 0" integrator="Euler"/><worldbody>'
    + "".join(
        f'<body name="slider{i}" pos="{position} 0 0">'
        f'<joint name="j{i}" type="slide" axis="1 0 0" damping=".1" armature=".01"/>'
        '<inertial pos="0 0 0" mass="1" diaginertia=".02 .025 .03"/>'
        '<geom type="box" size=".05 .04 .03" contype="0" conaffinity="0"/></body>'
        for i, position in enumerate(("-0.3", "0.3"))
    )
    + "</worldbody><actuator>"
    + "".join(
        f'<motor joint="j{i}" gear="1" ctrllimited="true" ctrlrange="-2 2" '
        'forcelimited="true" forcerange="-2 2"/>'
        for i in range(2)
    )
    + "</actuator></mujoco>"
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def encode(value):
    if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, "object array in evidence")
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "bytes_b64": base64.b64encode(value.tobytes()).decode(),
        }
    if isinstance(value, np.generic):
        return encode(value.item())
    if is_dataclass(value):
        return {
            "type": type(value).__module__ + "." + type(value).__name__,
            "fields": {f.name: encode(getattr(value, f.name)) for f in fields(value)},
        }
    if isinstance(value, Mapping):
        # Stable typed-key ordering for exact comparison, with all fields retained.
        entries = [[encode(key), encode(item)] for key, item in value.items()]
        entries.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True))
        return {"mapping_entries": entries}
    if isinstance(value, tuple):
        return {"tuple": [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_b64": base64.b64encode(value).decode()}
    if type(value) is float:
        return {"float64_bits": struct.pack(">d", value).hex()}
    if value is None or type(value) in (str, bool, int):
        return value
    raise TypeError("unsupported evidence value " + type(value).__name__)


def exact(left, right, label):
    require(encode(left) == encode(right), label)


class Journal:
    def __init__(self, path):
        self.path = path
        self.stream = gzip.open(path, "xb")
        self.sha, self.size, self.events = hashlib.sha256(), 0, 0

    def emit(self, event, **payload):
        line = (
            json.dumps(encode({"event": event, **payload}), allow_nan=False, separators=(",", ":"))
            + "\n"
        ).encode()
        self.stream.write(line)
        self.stream.flush()
        self.sha.update(line)
        self.size += len(line)
        self.events += 1

    def close(self):
        self.stream.close()
        with self.path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "path": str(self.path),
            "events": self.events,
            "logical_bytes": self.size,
            "logical_sha256": self.sha.hexdigest(),
            "compressed_bytes": self.path.stat().st_size,
            "compressed_sha256": digest,
        }


def provenance(root, *, preflight=False):
    def source_entry(path):
        return ((path / "pyproject.toml").is_file() and (path / "src/phymeo").is_dir()) or (
            path.name.lower() == "src"
            and (path / "phymeo").is_dir()
            and (path.parent / "pyproject.toml").is_file()
        )

    require((root / "phymeo/__init__.py").is_file(), "actual installed package required")
    search = [Path(item or Path.cwd()).resolve() for item in sys.path]
    require(not any(source_entry(path) for path in search), "checkout present in sys.path")
    if preflight:
        for path in (Path.cwd().resolve(), Path(__file__).resolve().parent):
            require(
                not any(source_entry(p) for p in (path, *path.parents)),
                "consumer/cwd must be outside checkout",
            )
    import phymeo

    require(Path(phymeo.__file__).resolve().is_relative_to(root), "installed product required")
    loaded = {}
    for name, module in sorted(sys.modules.items()):
        if (name == "phymeo" or name.startswith("phymeo.")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve(strict=True)
            require(path.is_relative_to(root), "wrong product origin: " + name)
            loaded[name] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    return {"installed_root": str(root), "cwd": str(Path.cwd().resolve()), "loaded": loaded}


def checkpoint_state(capsule):
    magic, digest, body = capsule.to_bytes().split(b"\n", 2)
    require(magic == b"PHYMEOTYXCP1", "known public capsule encoding")
    require(hashlib.sha256(body).hexdigest().encode() == digest, "capsule SHA")
    return json.loads(body)["state"]


def raw_row(record, index):
    values = np.frombuffer(base64.b64decode(record["data"]), dtype=record["$array"])
    return values.reshape(record["shape"])[index].tobytes()


def comparable(transition):
    result = {field.name: getattr(transition, field.name) for field in fields(transition)}
    for name in ("input_frame", "output_frame"):
        frame = result[name]
        result[name] = {"binding": frame.binding, "ticks": frame.ticks, "times": frame.times}
    result["action_receipt"] = {
        "binding": {
            key: value for key, value in transition.action_receipt.binding.items() if key != "epoch"
        },
        "controls": transition.action_receipt.controls,
    }
    return result


def run(output, journal, root):
    import mujoco as mj
    import torch

    from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState, MuJoCoRuntime
    from phymeo.backends.mujoco_task_workspace import MuJoCoTyxTaskSpec
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxActionSpec, MuJoCoTyxVectorEnv
    from phymeo.training.mujoco_tyx_reset import MuJoCoTyxResetPlan

    require(mj.__version__ == "3.3.7", "fixed MuJoCo compiler")
    require(torch.version.cuda is not None and torch.cuda.is_available(), "CUDA Torch required")
    torch.cuda.set_device(0)
    require(provenance(root)["loaded"], "product provenance")
    initial = {
        key: MuJoCoInitialState([sign * 0.06, -sign * 0.03], [rate * 0.04, -rate * 0.02])
        for key, (sign, rate) in zip(KEYS, ((1, 1), (1, -1), (-1, 1), (-1, -1)), strict=True)
    }
    compiled = compile_mujoco_xml(XML, require_asset=False)
    plan = MuJoCoTyxResetPlan(
        compiled,
        initial,
        q_indices=(0, 1),
        q_bounds=((-0.1, 0.1),) * 2,
        v_indices=(0, 1),
        v_bounds=((-0.04, 0.04),) * 2,
        seed=6511,
        namespace="N5",
    )
    task = MuJoCoTyxTaskSpec(
        (0, 1),
        (0, 1),
        scale=(5, 5, 1, 1),
        bias=(0, 0, 0, 0),
        target=(0, 0, 0, 0),
        weight=(12.5, 12.5, 0.005, 0.005),
        lower=(-1, -1, -8, -8),
        upper=(1, 1, 8, 8),
    )
    action = MuJoCoTyxActionSpec((0, 1), scale=(2, 2), bias=(0, 0))
    calls = 0

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU integration is forbidden for this TYX consumer")

    with ExitStack() as owners:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            owners.enter_context(patch.object(mj, name, forbidden))
        runtime = owners.enter_context(
            MuJoCoRuntime(compiled, mode="cuda", world_keys=KEYS, initial_states=initial)
        )
        env = owners.enter_context(
            MuJoCoTyxVectorEnv(runtime, task, action, h=H, horizons=(64,) * 4, reset_plan=plan)
        )
        raw_action = torch.tensor(
            [[0.1, -0.08], [-0.07, 0.05], [0.06, 0.03], [-0.04, -0.09]],
            dtype=torch.float32,
            device=env.device,
        )

        def advance(label):
            nonlocal calls
            entry = runtime.snapshot()
            journal.emit(
                "step-request", label=label, entry=entry, action=raw_action, frame=env.current_frame
            )
            transition = env.step(raw_action, world_keys=KEYS, expected_frame=env.current_frame)
            calls += 1
            journal.emit(
                "step-result",
                label=label,
                transition=transition,
                random_reset=env.capture_training_state()["random_reset"],
            )
            require(
                dict(transition.accepted_worlds) == dict.fromkeys(KEYS, True),
                "every declared world must accept",
            )
            require(
                not bool(transition.final_mask.any()), "no automatic reset in this short window"
            )
            dt = float(np.float32(H))
            for key in KEYS:
                state = transition.states[key]
                force = state.last_step.stored_control.ctrl
                velocity = (1.01 * entry[key].v + dt * force) / (1.01 + dt * 0.1)
                position = entry[key].q + dt * velocity
                np.testing.assert_allclose(state.v, velocity, rtol=0, atol=3e-3)
                np.testing.assert_allclose(state.q, position, rtol=0, atol=2e-4)
                require(state.tick == entry[key].tick + 1, "real physical tick")
                require(not np.array_equal(state.q, entry[key].q), "real motion required")
            journal.emit("step-checked", label=label, independent_qv_recurrence=True)
            return transition

        try:
            journal.emit(
                "initial",
                binding=env.training_binding(),
                source=compiled.source,
                plan=plan.binding,
                initial=runtime.snapshot(),
                modules=provenance(root),
            )
            advance("warm-0")
            advance("warm-1")
            before = runtime.checkpoint()
            state_before = runtime.snapshot()
            proposal = plan.prepare((103, 7), env.reset_indices)
            env.reset((103, 7))
            receipt = env.last_reset_receipt
            exact(receipt.requested, proposal.requested, "public sampled requests")
            exact(dict(env.reset_indices), dict(proposal.next_indices), "committed stream indices")
            after = runtime.checkpoint()
            raw_before, raw_after = checkpoint_state(before), checkpoint_state(after)
            for index in (1, 3):
                require(
                    raw_before["worlds"][index] == raw_after["worlds"][index],
                    "unselected complete accepted mirror",
                )
                for name in ("q", "v", "clock"):
                    require(
                        raw_row(raw_before[name], index) == raw_row(raw_after[name], index),
                        "unselected raw " + name,
                    )
                exact(
                    runtime.snapshot()[KEYS[index]],
                    state_before[KEYS[index]],
                    "unselected complete observation",
                )
            journal.emit(
                "partial-reset",
                receipt=receipt,
                proposed=proposal,
                before=before.to_bytes(),
                after=after.to_bytes(),
            )
            advance("partial-successor-0")
            advance("partial-successor-1")
            capsule = runtime.checkpoint()
            path = output / "accepted.checkpoint"
            path.write_bytes(capsule.to_bytes())
            capsule = MuJoCoTyxCheckpoint.from_bytes(path.read_bytes())
            saved_env, saved_physics = env.capture_training_state(), runtime.snapshot()
            predicted = plan.prepare((4096, 31), env.reset_indices)
            journal.emit(
                "composition-save",
                capsule=capsule.to_bytes(),
                env=saved_env,
                physics=saved_physics,
                next_sample=predicted,
            )

            env.reset((4096, 31))
            receipt_a = env.last_reset_receipt
            exact(receipt_a.requested, predicted.requested, "next sample A")
            first = (advance("A-0"), advance("A-1"))
            final_a, physics_a = env.capture_training_state(), runtime.snapshot()

            restored = env.restore_training_state(
                capsule, saved_env, expected_revision=runtime.checkpoint_revision
            )
            exact(env.capture_training_state(), saved_env, "full env state restored")
            exact(runtime.snapshot(), saved_physics, "full physics restored")
            require(env.last_reset_receipt is None, "restore is not a fabricated reset")
            env.reset((31, 4096))
            receipt_b = env.last_reset_receipt
            exact(
                receipt_b.requested, predicted.requested, "next sample independent of order/restore"
            )
            exact(receipt_b.stored, receipt_a.stored, "actual stored reset states")
            second = (advance("B-0"), advance("B-1"))
            exact(
                tuple(map(comparable, first)),
                tuple(map(comparable, second)),
                "all successor transition fields except monotonic publications",
            )
            exact(env.capture_training_state(), final_a, "episode/action/stream state continuation")
            exact(runtime.snapshot(), physics_a, "complete physical continuation")
            require(calls == 8, "fixed eight batch steps")
            journal.emit(
                "composition-restored",
                receipt=restored,
                reset_a=receipt_a,
                reset_b=receipt_b,
                final=runtime.snapshot(),
                env=env.capture_training_state(),
            )
            result = {
                "batch_steps": calls,
                "world_steps": calls * len(KEYS),
                "reset_indices": dict(env.reset_indices),
                "plan_sha256": plan.sha256,
                "checkpoint_sha256": capsule.sha256,
                "counters": dict(runtime.counters),
                "modules": provenance(root),
            }
        finally:
            journal.emit(
                "before-owner-close",
                calls=calls,
                counters=dict(runtime.counters),
                needs_recovery=env.needs_recovery,
                reset_indices=dict(env.reset_indices),
            )
    journal.emit("owners-closed", calls=calls)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cuda",), default="cuda")
    parser.add_argument("--installed-root", "--expected-install-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.output.is_absolute() and args.installed_root.is_absolute(), "absolute paths required"
    )
    root = args.installed_root.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "random-reset.jsonl.gz")
    summary = {
        "status": "RUNNING",
        "mode": args.mode,
        "consumer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "installed J W4 public random reset and exact Env/physics composition",
        "excluded": "PPO learning, model randomization, root/ball sampling, full-GPU reset",
    }
    try:
        summary["initial_provenance"] = provenance(root, preflight=True)
        summary.update(run(output, journal, root))
        summary["status"] = "PASS"
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        journal.emit("failure", summary=summary)
        raise
    finally:
        try:
            summary["final_provenance"] = provenance(root)
        except BaseException:
            summary["status"] = "FAIL"
            summary["provenance_error"] = traceback.format_exc()
            raise
        finally:
            journal.emit("summary", summary=summary)
            summary["trace"] = journal.close()
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
            )
            print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()

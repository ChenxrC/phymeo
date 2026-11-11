"""Installed N6 control windows and a bounded healthy GRU continuation.

Copy outside every checkout. Inline J/S source, task/action definitions and N5
reset seed match the frozen public examples. No source tests/examples imports,
CPU stepping, learning-success claim or extra optimization budget.
"""

import argparse
import base64
import gzip
import hashlib
import json
import random
import struct
import sys
import traceback
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

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
            json.dumps(
                encode({"event": event, "record_ordinal": self.events, **payload}),
                allow_nan=False,
                separators=(",", ":"),
            )
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
        require(
            not any(name == "phymeo" or name.startswith("phymeo.") for name in sys.modules),
            "product must not be imported before provenance preflight",
        )
        for path in (Path.cwd().resolve(), Path(__file__).resolve().parent, root):
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


SUPPORT_XML = """<mujoco model="t3-normal"><compiler angle="radian" fusestatic="false"/>
<option timestep=".001" gravity="0 0 -9.81" integrator="Euler" solver="Newton"
 iterations="128" tolerance="1e-10" jacobian="dense"/>
<default><geom condim="1" solref=".02 1" solimp=".8 .9 .002 .5 2"/>
<joint armature=".01" damping=".04"/></default>
<worldbody><geom name="platform" type="box" size=".24 .18 .05"/>
<body name="root" pos="0 0 .109"><joint name="vertical" type="slide" axis="0 0 1" armature="0" damping="0"/>
<inertial mass="2.0" pos="0 0 0" fullinertia=".08 .10 .12 .002 -.001 .003"/>
<geom name="moving" type="box" size=".12 .08 .06"/></body></worldbody>
<actuator><motor joint="vertical"/></actuator></mujoco>"""


KEYS = (7, 31, 103, 4096)
HORIZONS = (3, 5, 7, 11)
SEED = 6511


def components(case):
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState
    from phymeo.backends.mujoco_task_workspace import MuJoCoTyxTaskSpec
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxActionSpec
    from phymeo.training.mujoco_tyx_reset import MuJoCoTyxResetPlan

    require(case in ("J", "S"), "fixed task identity")
    if case == "J":
        xml, h, width = XML, 0.002, 2
        target, qs, vs = (0.0, 0.0), (0.2, 0.2), (1.0, 1.0)
        lower, upper, vbound, action_scale = (-1.0, -1.0), (1.0, 1.0), 8.0, 2.0
        qbounds, vbounds = ((-0.1, 0.1),) * 2, ((-0.04, 0.04),) * 2
        initial = [
            ((s * 0.06, -s * 0.03), (t * 0.04, -t * 0.02))
            for s, t in ((1, 1), (1, -1), (-1, 1), (-1, -1))
        ]
    else:
        xml, h, width = SUPPORT_XML, 0.001, 1
        target, qs, vs = (0.00075,), (0.0005,), (0.05,)
        lower, upper, vbound, action_scale = (-0.002,), (0.004,), 0.5, 20.0
        qbounds, vbounds = ((0.00065, 0.00085),), ((-0.002, 0.002),)
        initial = [
            ((0.00075 + s * 0.00010,), (t * 0.002,))
            for s, t in ((1, 1), (1, -1), (-1, 1), (-1, -1))
        ]
    compiled = compile_mujoco_xml(xml, require_asset=False)
    anchors = {key: MuJoCoInitialState(*row) for key, row in zip(KEYS, initial, strict=True)}
    plan = MuJoCoTyxResetPlan(
        compiled,
        anchors,
        q_indices=tuple(range(width)),
        q_bounds=qbounds,
        v_indices=tuple(range(width)),
        v_bounds=vbounds,
        seed=SEED,
        namespace="N6",
    )
    task = MuJoCoTyxTaskSpec(
        tuple(range(width)),
        tuple(range(width)),
        scale=tuple(1 / x for x in qs) + tuple(1 / x for x in vs),
        bias=tuple(-q / x for q, x in zip(target, qs, strict=True)) + (0.0,) * width,
        target=target + (0.0,) * width,
        weight=tuple(1 / (width * x * x) for x in qs) + tuple(0.01 / (width * x * x) for x in vs),
        lower=lower + (-vbound,) * width,
        upper=upper + (vbound,) * width,
    )
    action = MuJoCoTyxActionSpec(
        tuple(range(width)), scale=[action_scale] * width, bias=[0.0] * width
    )
    return dict(
        compilation=compiled,
        anchors=anchors,
        reset_plan=plan,
        task_spec=task,
        action_spec=action,
        h=h,
        xml=xml,
    )


@contextmanager
def environment(case, decimation):
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.training.mujoco_tyx_control_window import MuJoCoTyxControlWindowSpec
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxVectorEnv

    parts = components(case)
    with ExitStack() as owners:
        runtime = owners.enter_context(
            MuJoCoRuntime(
                parts["compilation"], mode="cuda", world_keys=KEYS, initial_states=parts["anchors"]
            )
        )
        env = owners.enter_context(
            MuJoCoTyxVectorEnv(
                runtime,
                parts["task_spec"],
                parts["action_spec"],
                h=parts["h"],
                horizons=HORIZONS,
                reset_plan=parts["reset_plan"],
                control_window=MuJoCoTyxControlWindowSpec(decimation),
            )
        )
        yield env, parts


def endpoint(transition, spec, expected_lengths):
    raw = np.array(
        [
            np.r_[
                transition.states[key].q[list(spec.q_indices)],
                transition.states[key].v[list(spec.v_indices)],
            ]
            for key in KEYS
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        transition.final_observation.cpu().numpy(),
        raw * spec.scale + spec.bias,
        rtol=2e-5,
        atol=2e-6,
    )
    reward = -np.sum(spec.weight * (raw - spec.target) ** 2, axis=1)
    np.testing.assert_allclose(transition.reward.cpu().numpy(), reward, rtol=2e-5, atol=2e-6)
    terminated = np.any((raw < spec.lower) | (raw > spec.upper), axis=1)
    timed = (expected_lengths >= np.asarray(HORIZONS)) & ~terminated
    np.testing.assert_array_equal(transition.terminated.cpu().numpy(), terminated)
    np.testing.assert_array_equal(transition.time_outs.cpu().numpy(), timed)
    np.testing.assert_array_equal(transition.final_mask.cpu().numpy(), terminated | timed)
    np.testing.assert_array_equal(transition.episode_length, expected_lengths)
    return terminated | timed


def physics(case, before, after, h):
    dt = float(np.float32(h))
    supported = 0
    for key in KEYS:
        state = after[key]
        receipt = state.last_step
        require(
            state.tick == before[key].tick + 1 and receipt.tick == state.tick,
            "one physical tick and receipt per substep",
        )
        require(
            receipt.input_time == before[key].time and receipt.output_time == state.time,
            "physical input/output clock binding",
        )
        control = receipt.stored_control
        if case == "J":
            force = np.asarray(control.ctrl) + np.asarray(control.qfrc_applied)
            force = force + np.asarray(control.xfrc_applied)[1:3, 0]
            velocity = (1.01 * before[key].v + dt * force) / (1.01 + 0.1 * dt)
        else:
            support = np.zeros(6)
            for contact in receipt.contacts:
                require(
                    contact.dimension == 1 and contact.normal_force >= 0,
                    "original S normal-only law",
                )
                force1 = np.asarray(contact.wrench_on_body1_world)[:3]
                force2 = np.asarray(contact.wrench_on_body2_world)[:3]
                np.testing.assert_allclose(force1 + force2, 0.0, rtol=0, atol=2e-4)
                if contact.body_ids[0] == 1:
                    support += np.asarray(contact.wrench_on_body1_world)
                if contact.body_ids[1] == 1:
                    support += np.asarray(contact.wrench_on_body2_world)
            require(
                np.all(np.isfinite(support)) and support[2] > 0,
                "real moving-root support in every S substep",
            )
            supported += 1
            external = float(control.ctrl[0] + control.qfrc_applied[0] + control.xfrc_applied[1, 2])
            # Exact ordinary 2 kg vertical slider: no root armature/damping.
            velocity = before[key].v + dt * (external + support[2] - 2 * 9.81) / 2
        np.testing.assert_allclose(state.v, velocity, rtol=0, atol=3e-3)
        np.testing.assert_allclose(state.q, before[key].q + dt * velocity, rtol=0, atol=2e-4)
    return supported


def final_evidence(journal, event, env, **payload):
    """A failed diagnostic must not replace an already active original error."""
    original_error = sys.exception()
    try:
        payload["current"] = env.runtime.snapshot()
        failure = env.last_failure
        payload.update(
            counters=dict(env.counters),
            needs_recovery=env.needs_recovery,
            failure=None if failure is None else repr(failure),
            failed_window=None if failure is None else failure.control_window,
        )
    except BaseException:
        if original_error is None:
            raise
        payload["final_snapshot_error"] = traceback.format_exc()
    try:
        journal.emit(event, **payload)
    except BaseException:
        if original_error is None:
            raise


def fixed_windows(case, decimation, journal):
    import torch

    with environment(case, decimation) as (env, parts):
        lengths = np.zeros(4, np.int64)
        support_count = reset_count = 0
        journal.emit(
            "window-initial",
            case=case,
            decimation=decimation,
            xml=parts["xml"],
            source=parts["compilation"].source,
            binding=env.training_binding(),
            reset_binding=parts["reset_plan"].binding,
            states=env.runtime.snapshot(),
        )
        try:
            for ordinal in range(4):
                before = env.runtime.snapshot()
                action = torch.full(
                    (4, env.num_actions),
                    (ordinal + 1) / 100,
                    dtype=torch.float32,
                    device=env.device,
                )
                journal.emit(
                    "window-request",
                    case=case,
                    ordinal=ordinal,
                    action=action,
                    entry=before,
                    frame=env.current_frame,
                )
                transition = env.step(action, world_keys=KEYS, expected_frame=env.current_frame)
                journal.emit(
                    "window-accepted",
                    case=case,
                    ordinal=ordinal,
                    transition=transition,
                    reset=env.last_reset_receipt,
                    counters=env.counters,
                    env_state=env.capture_training_state(),
                )
                window = transition.control_window
                require(
                    window.physics_steps == decimation
                    and window.completed == decimation
                    and window.attempted == decimation
                    and window.phase == "complete"
                    and window.failed_substep is None,
                    "complete ordered window",
                )
                require(
                    tuple(transition.world_keys) == KEYS
                    and all(transition.accepted_worlds.values()),
                    "complete world membership",
                )
                require(
                    env.decimation == decimation and env.policy_dt_s == decimation * env.h,
                    "physical/policy rates",
                )
                exact(
                    window.action_receipt, transition.action_receipt, "single held action receipt"
                )
                for subordinal, sub in enumerate(window.substeps):
                    require(
                        sub.ordinal == subordinal
                        and sub.diagnostics_complete
                        and dict(sub.accepted_worlds) == dict.fromkeys(KEYS, True),
                        "ordered accepted substep diagnostics",
                    )
                    for key in KEYS:
                        exact(
                            sub.states[key].last_step.stored_control,
                            transition.action_receipt.controls[key],
                            "held channels",
                        )
                    support_count += physics(case, before, sub.states, env.h)
                    before = sub.states
                exact(transition.states, window.substeps[-1].states, "endpoint uses final physics")
                lengths += 1
                done = endpoint(transition, env.task_spec, lengths)
                selected = tuple(key for key, flag in zip(KEYS, done, strict=True) if flag)
                if selected:
                    require(
                        env.last_reset_receipt.world_keys == selected,
                        "endpoint partial reset owners",
                    )
                    require(len(selected) < len(KEYS), "exercise a genuine partial reset")
                    for key, state in env.runtime.snapshot().items():
                        if key not in selected:
                            exact(
                                state, transition.states[key], "unselected world unchanged by reset"
                            )
                    reset_count += len(selected)
                    lengths[done] = 0
                np.testing.assert_array_equal(env.episode_length, lengths)
            require(
                env.counters["transitions"] == 4
                and env.counters["physical_steps"] == 4 * decimation,
                "separate policy and physical counters",
            )
            require(
                env.runtime.counters["device_control_submissions"] == 4,
                "exactly one device action submission per policy window",
            )
            require(reset_count > 0, "fixed horizons exercise N5 reset")
            if case == "S":
                require(support_count == 4 * 4 * decimation, "continuous S support")
            result = dict(
                case=case,
                decimation=decimation,
                policy_windows=4,
                physical_batch_steps=4 * decimation,
                physical_world_steps=16 * decimation,
                support_world_substeps=support_count,
                reset_count=reset_count,
                counters=dict(env.counters),
                binding=env.training_binding(),
            )
            journal.emit("window-checked", result=result)
            return result
        finally:
            final_evidence(journal, "window-final", env, case=case)


def semantic_transition(transition):
    # Restore cannot rewind publication/CAS identities. Keep the full unmodified
    # record in the journal; only these named identities are excluded here.
    result = {item.name: getattr(transition, item.name) for item in fields(transition)}
    for name in ("input_frame", "output_frame"):
        frame = result[name]
        result[name] = dict(binding=frame.binding, ticks=frame.ticks, times=frame.times)
    receipt = transition.action_receipt
    result["action_receipt"] = dict(
        binding={key: value for key, value in receipt.binding.items() if key != "epoch"},
        controls=receipt.controls,
    )
    window = transition.control_window
    result["control_window"] = dict(
        spec_sha256=window.spec_sha256,
        physics_steps=window.physics_steps,
        attempted=window.attempted,
        completed=window.completed,
        phase=window.phase,
        failed_substep=window.failed_substep,
        action_receipt=result["action_receipt"],
        substeps=tuple(
            dict(
                ordinal=sub.ordinal,
                accepted_worlds=sub.accepted_worlds,
                states=sub.states,
                diagnostics_complete=sub.diagnostics_complete,
                error=sub.error,
            )
            for sub in window.substeps
        ),
    )
    return result


def mask_checks(record):
    import torch

    partial = 0
    for entry in record["recurrent_masks"]:
        done = entry["done"]
        partial += int(bool(done.any()) and not bool(done.all()))
        for name in ("actor", "critic"):
            before, after = entry["before"][name], entry["after"][name]
            require(
                torch.equal(before[:, ~done], after[:, ~done]),
                "non-done GRU rows unchanged by mask",
            )
            require(not bool(torch.count_nonzero(after[:, done])), "done GRU rows cleared")
    return partial


def continuation(output, journal):
    import torch

    from phymeo.training.mujoco_tyx_checkpoint import MuJoCoTyxPpoSession
    from phymeo.training.rsl_mujoco_tyx import RslMuJoCoTyxVecEnv

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with environment("J", 2) as (env, parts):
        try:
            env.reset()
            session = MuJoCoTyxPpoSession(RslMuJoCoTyxVecEnv(env), steps=4, recurrent=True)
            original_step = session.adapter.step
            update_phase = ["warm"]
            attempted_windows = [0]

            def observed_step(actions):
                ordinal = attempted_windows[0]
                attempted_windows[0] += 1
                journal.emit(
                    "gru-window-request",
                    update=update_phase[0],
                    ordinal=ordinal,
                    actions=actions,
                    frame=env.current_frame,
                )
                result = original_step(actions)
                # Persist each accepted transition before the next policy window
                # or any fallible optimization; never repeat physical work.
                journal.emit(
                    "gru-window-accepted",
                    update=update_phase[0],
                    ordinal=ordinal,
                    transition=session.adapter.last_transition,
                )
                return result

            session.adapter.step = observed_step
            journal.emit(
                "gru-initial",
                xml=parts["xml"],
                source=parts["compilation"].source,
                binding=env.training_binding(),
                training=session.training_state(),
                physics=env.runtime.snapshot(),
                reset=env.last_reset_receipt,
            )
            warm = session.update()
            journal.emit(
                "gru-warm",
                record=warm,
                training=session.training_state(),
                physics=env.runtime.snapshot(),
                counters=env.counters,
            )
            require(
                all(
                    bool(torch.count_nonzero(value))
                    for value in session.training_state()["hidden"].values()
                ),
                "actual nonzero actor and critic GRU hidden state",
            )
            checkpoint = output / "J-d2-gru-checkpoint"
            saved = session.save_checkpoint(checkpoint)
            saved_training = session.training_state()
            saved_physics = env.runtime.snapshot()
            journal.emit(
                "gru-save",
                receipt=saved,
                training=saved_training,
                physics=saved_physics,
                physical_capsule=env.runtime.checkpoint().to_bytes(),
            )
            update_phase[0] = "A"
            a = session.update()
            train_a, physical_a = session.training_state(), env.runtime.snapshot()
            journal.emit(
                "gru-A", record=a, training=train_a, physics=physical_a, counters=env.counters
            )
            env.reset((103, 7))
            next_a = (env.runtime.snapshot(), env.capture_training_state()["random_reset"])
            journal.emit("gru-next-reset-A", receipt=env.last_reset_receipt, next_state=next_a)
            restored = session.restore_checkpoint(
                checkpoint, expected_revision=env.runtime.checkpoint_revision
            )
            journal.emit(
                "gru-restore",
                receipt=restored,
                training=session.training_state(),
                physics=env.runtime.snapshot(),
                counters=env.counters,
            )
            exact(session.training_state(), saved_training, "healthy complete training restoration")
            exact(env.runtime.snapshot(), saved_physics, "healthy complete physics restoration")
            update_phase[0] = "B"
            b = session.update()
            train_b, physical_b = session.training_state(), env.runtime.snapshot()
            journal.emit(
                "gru-B", record=b, training=train_b, physics=physical_b, counters=env.counters
            )
            exact(train_a, train_b, "learner/Adam/RNG/GRU/env continuation")
            exact(physical_a, physical_b, "complete physical continuation")
            exact(
                tuple(map(semantic_transition, a["transitions"])),
                tuple(map(semantic_transition, b["transitions"])),
                "substep receipt and endpoint transition continuation",
            )
            exact(a["loss"], b["loss"], "same upstream update loss")
            exact(a["optimization"], b["optimization"], "same upstream optimization evidence")
            env.reset((7, 103))
            next_b = (env.runtime.snapshot(), env.capture_training_state()["random_reset"])
            journal.emit("gru-next-reset-B", receipt=env.last_reset_receipt, next_state=next_b)
            exact(next_a, next_b, "next sample independent of restore and subset order")
            masks = sum(mask_checks(record) for record in (warm, a, b))
            require(masks > 0, "actual recurrent partial-reset masks")
            for record in (warm, a, b):
                require(len(record["transitions"]) == 4, "exactly four rollout windows per update")
                for transition in record["transitions"]:
                    window = transition.control_window
                    require(
                        window.completed == 2
                        and window.attempted == 2
                        and window.failed_substep is None
                        and window.phase == "complete",
                        "GRU uses successful D2 physical windows",
                    )
            require(
                env.counters["transitions"] == 12 and env.counters["physical_steps"] == 24,
                "warm/A/B only: twelve policy windows and twenty-four physical steps",
            )
            require(
                env.runtime.counters["device_control_submissions"] == 12,
                "GRU one action submission per window",
            )
            with environment("J", 4) as (wrong_env, unused_parts):
                del unused_parts
                other = MuJoCoTyxPpoSession(RslMuJoCoTyxVecEnv(wrong_env), steps=4, recurrent=True)
                old_capsule = wrong_env.runtime.checkpoint().to_bytes()
                old_training = other.training_state()
                try:
                    other.restore_checkpoint(
                        checkpoint, expected_revision=wrong_env.runtime.checkpoint_revision
                    )
                except (ValueError, RuntimeError) as error:
                    require("binding" in str(error), "cross-D binding rejection")
                    journal.emit(
                        "cross-D-rejected",
                        error=str(error),
                        binding=wrong_env.training_binding(),
                        before_physics=old_capsule,
                        after_physics=wrong_env.runtime.checkpoint().to_bytes(),
                        before_training=old_training,
                        after_training=other.training_state(),
                    )
                else:
                    raise AssertionError("cross-decimation checkpoint restore accepted")
                require(
                    wrong_env.runtime.checkpoint().to_bytes() == old_capsule,
                    "cross-D rejection leaves physical capsule unchanged",
                )
                exact(
                    other.training_state(),
                    old_training,
                    "cross-D rejection leaves learner and RNG unchanged",
                )
                require(
                    wrong_env.counters["transitions"] == 0
                    and wrong_env.counters["physical_steps"] == 0,
                    "cross-D guard advances no physics",
                )
            result = dict(
                case="J",
                decimation=2,
                rollout_steps=4,
                updates_executed=3,
                policy_windows=12,
                physical_batch_steps=24,
                physical_world_steps=96,
                partial_masks=masks,
                cross_decimation_refused=True,
                next_reset_equal=True,
                save=saved,
                counters=dict(env.counters),
                semantic_exclusions=(
                    "frame publication revision/epoch",
                    "action publication epoch/revision",
                    "substep entry/exit revision",
                ),
            )
            journal.emit("gru-checked", result=result)
            return result
        finally:
            final_evidence(journal, "gru-final", env)


def run(output, journal, root):
    import mujoco as mj
    import torch

    require(mj.__version__ == "3.3.7", "fixed native compiler version")
    require(torch.version.cuda is not None and torch.cuda.is_available(), "CUDA Torch required")
    torch.cuda.set_device(0)
    forbidden_calls = []

    def forbidden(*args, **kwargs):
        forbidden_calls.append("CPU integration attempted")
        raise AssertionError("CPU step/step1/step2 cannot advance MuJoCo-TYX")

    with ExitStack() as guards:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            guards.enter_context(patch.object(mj, name, forbidden))
        first = fixed_windows("J", 2, journal)
        second = fixed_windows("S", 4, journal)
        resumed = continuation(output, journal)
    require(not forbidden_calls, "no CPU integration fallback")
    modules = provenance(root)
    require(
        "phymeo.training.mujoco_tyx_control_window" in modules["loaded"],
        "installed N6 specification",
    )
    require(
        "phymeo.training.mujoco_tyx_checkpoint" in modules["loaded"], "installed N3 composition"
    )
    journal.emit(
        "all-checked",
        windows=(first, second),
        continuation=resumed,
        forbidden_cpu_calls=forbidden_calls,
        modules=modules,
    )
    return dict(
        windows=(first, second),
        continuation=resumed,
        cpu_integration_calls=0,
        policy_windows=20,
        physical_batch_steps=48,
        physical_world_steps=192,
        final_provenance=modules,
    )


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
    journal = Journal(output / "control-window.jsonl.gz")
    summary = dict(
        status="RUNNING",
        mode=args.mode,
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="installed N6 J D2 and S D4, N5 partial reset, healthy N3 GRU continuation",
        excluded="learned-policy success, longer training, fault recovery, source suite rerun",
    )
    try:
        summary["initial_provenance"] = provenance(root, preflight=True)
        journal.emit("initial-provenance", value=summary["initial_provenance"])
        result = run(output, journal, root)
        summary["result"] = encode(result)
        for key in (
            "policy_windows",
            "physical_batch_steps",
            "physical_world_steps",
            "cpu_integration_calls",
        ):
            summary[key] = result[key]
        summary["status"] = "PASS"
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["exception"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
        )
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

"""Short public endpoint control windows and exact GRU continuation.

No learning claim: the original J/S models, task definitions and action maps
are reused. One device action drives D original physical steps. HOST reference
input and public state observation are explicit; no CPU dynamics implements TYX.
"""

import argparse
import importlib.util
import json
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import fields
from pathlib import Path

import numpy as np

KEYS = (7, 31, 103, 4096)
HORIZONS = (3, 5, 7, 11)
_SOURCE = None


def source():
    global _SOURCE
    if _SOURCE is None:
        spec = importlib.util.spec_from_file_location(
            "n6_original_random_reset_components",
            Path(__file__).with_name("mujoco_tyx_random_reset.py"),
        )
        _SOURCE = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _SOURCE
        spec.loader.exec_module(_SOURCE)
    return _SOURCE


def require_cuda():
    return source().require_cuda()


def json_text(value):
    return source().json_text(value)


@contextmanager
def make_environment(case, physics_steps=1, *, horizons=HORIZONS, task_spec=None):
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.training.mujoco_tyx_control_window import MuJoCoTyxControlWindowSpec
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxVectorEnv

    require_cuda()
    parts = source().components(case, keys=KEYS, seed=6511, namespace="N6")
    options = {} if physics_steps is None else {"control_window": MuJoCoTyxControlWindowSpec(physics_steps)}
    with ExitStack() as owners:
        runtime = owners.enter_context(
            MuJoCoRuntime(parts["compilation"], mode="cuda", world_keys=KEYS, initial_states=parts["anchors"])
        )
        env = owners.enter_context(
            MuJoCoTyxVectorEnv(
                runtime,
                parts["task_spec"] if task_spec is None else task_spec,
                parts["action_spec"],
                h=parts["h"],
                horizons=horizons,
                reset_plan=parts["reset_plan"],
                **options,
            )
        )
        yield env


def _assert_same(actual, expected, label):
    if json_text(actual) != json_text(expected):
        raise AssertionError(label)


def _endpoint_oracle(transition, spec):
    raw = np.array([
        np.r_[state.q[list(spec.q_indices)], state.v[list(spec.v_indices)]]
        for state in transition.states.values()
    ], dtype=np.float32)
    np.testing.assert_allclose(
        transition.final_observation.cpu().numpy(), raw * spec.scale + spec.bias,
        rtol=2e-5, atol=2e-6,
    )
    reward = -np.sum(spec.weight * (raw - spec.target) ** 2, axis=1)
    np.testing.assert_allclose(transition.reward.cpu().numpy(), reward, rtol=2e-5, atol=2e-6)
    terminated = np.any((raw < spec.lower) | (raw > spec.upper), axis=1)
    np.testing.assert_array_equal(transition.terminated.cpu().numpy(), terminated)


def _joint_recurrence(before, after, h):
    # Independent ordinary two-slider Euler equation, using the actual stored input.
    dt = float(np.float32(h))
    for key, state in after.items():
        packet = state.last_step.stored_control
        force = np.asarray(packet.ctrl) + np.asarray(packet.qfrc_applied)
        force = force + np.asarray(packet.xfrc_applied)[1:3, 0]
        velocity = (1.01 * before[key].v + dt * force) / (1.01 + 0.1 * dt)
        position = before[key].q + dt * velocity
        np.testing.assert_allclose(state.v, velocity, rtol=0, atol=3e-3)
        np.testing.assert_allclose(state.q, position, rtol=0, atol=2e-4)


def run_case(case, physics_steps, output):
    """Exactly eight policy windows; complete substep and reference evidence."""
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime

    if (case, physics_steps) not in (("J", 1), ("J", 2), ("J", 4), ("S", 4)):
        raise ValueError("fixed window matrix is J D1/2/4 and S D4")
    torch = require_cuda()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = f"{case}-d{physics_steps}"
    summary = {"status": "FAILED_CONTROL_WINDOW", "case": case, "decimation": physics_steps}
    parts = source().components(case, keys=KEYS, seed=6511, namespace="N6")
    with (output / (name + ".jsonl")).open("x", encoding="utf-8") as log:
        with ExitStack() as owners:
            env = owners.enter_context(make_environment(case, physics_steps))
            reference = owners.enter_context(MuJoCoRuntime(
                parts["compilation"], mode="cuda", world_keys=KEYS, initial_states=parts["anchors"]
            ))
            contact_count = 0
            try:
                log.write(json_text({"event": "initial", "binding": env.training_binding(), "states": env.runtime.snapshot(), "source": parts["compilation"].source.archive_sha256}) + "\n")
                for policy_step in range(8):
                    action = torch.full((4, env.num_actions), (policy_step + 1) / 100, device=env.device)
                    log.write(json_text({"event": "request", "ordinal": policy_step, "action": action}) + "\n")
                    transition = env.step(action, world_keys=KEYS, expected_frame=env.current_frame)
                    # Save accepted work before any independent comparison can fail.
                    log.write(json_text({"event": "transition", "ordinal": policy_step, "value": transition, "reset": env.last_reset_receipt}) + "\n")
                    window = transition.control_window
                    if not all(transition.accepted_worlds.values()) or window.completed != physics_steps or window.attempted != physics_steps:
                        raise AssertionError("incomplete physical control window")
                    if env.decimation != physics_steps or env.policy_dt_s != physics_steps * env.h:
                        raise AssertionError("public frequency metadata differs")
                    reference.set_controls(transition.action_receipt.controls)
                    previous = reference.snapshot()
                    reference_states = []
                    for ordinal, sub in enumerate(window.substeps):
                        if sub.ordinal != ordinal or not sub.diagnostics_complete or not all(sub.accepted_worlds.values()):
                            raise AssertionError("incomplete ordered substep evidence")
                        if not all(reference.step(env.h).values()):
                            raise AssertionError("independent held-input physical step refused")
                        expected = reference.snapshot()
                        reference_states.append(expected)
                        _assert_same(sub.states, expected, "complete substep physics differs from held HOST route")
                        for key, state in sub.states.items():
                            if state.tick != previous[key].tick + 1 or state.last_step.tick != state.tick:
                                raise AssertionError("physical tick/receipt does not advance once")
                            if state.time != state.last_step.output_time or state.last_step.input_time != previous[key].time:
                                raise AssertionError("actual physical clock/receipt differs")
                            _assert_same(state.last_step.stored_control, transition.action_receipt.controls[key], "held native channels changed")
                            contact_count += int(any(c.normal_force > 0 for c in state.last_step.contacts))
                        if case == "J":
                            _joint_recurrence(previous, sub.states, env.h)
                        previous = expected
                    log.write(json_text({"event": "reference", "ordinal": policy_step, "substeps": reference_states}) + "\n")
                    _assert_same(transition.states, window.substeps[-1].states, "transition is not last physical endpoint")
                    _endpoint_oracle(transition, env.task_spec)
                    if not np.all(transition.episode_length >= 1):
                        raise AssertionError("episode counter did not advance")
                    selected = tuple(key for key, flag in zip(KEYS, transition.final_mask.cpu().tolist(), strict=True) if flag)
                    if selected:
                        receipt = env.last_reset_receipt
                        if receipt.world_keys != selected:
                            raise AssertionError("endpoint subset reset identity differs")
                        reference.reset_states(receipt.stored, expected_revision=reference.checkpoint_revision)
                    _assert_same(env.runtime.snapshot(), reference.snapshot(), "post-window/reset state differs")
                if case == "S" and contact_count != 4 * 8 * physics_steps:
                    raise AssertionError("continuous support missing an actual substep contact")
                submissions = env.runtime.counters["device_control_submissions"]
                if submissions != 8 or env.counters["transitions"] != 8 or env.counters["physical_steps"] != 8 * physics_steps:
                    raise AssertionError("physical/policy/action counter scopes differ")
                consumed = any(env.reset_indices.values())
                if not consumed:
                    raise AssertionError("fixed horizon did not exercise random reset")
                summary.update(
                    status="PASS_CONTROL_WINDOW", policy_steps=8, physical_steps=8 * physics_steps,
                    compared_world_substeps=4 * 8 * physics_steps, action_submissions=submissions,
                    contact_world_substeps=contact_count, random_reset_consumed=consumed,
                    reset_indices=dict(env.reset_indices), binding=env.training_binding(),
                    physics_and_policy_rates={"physics_hz": 1 / env.h, "policy_hz": 1 / env.policy_dt_s},
                )
            except BaseException as error:
                failure = env.last_failure
                log.write(json_text({"event": "failure", "error": repr(error), "needs_recovery": env.needs_recovery, "control_window": None if failure is None else failure.control_window, "counters": env.counters}) + "\n")
                raise
            finally:
                (output / (name + "-summary.json")).write_text(json_text(summary), encoding="utf-8")
    return summary


def _semantic_transition(transition):
    # Restore retains monotonic publication identities, never rewinds them.
    result = source()._comparable_transition(transition)
    window = transition.control_window
    result["control_window"] = {
        "spec_sha256": window.spec_sha256, "physics_steps": window.physics_steps,
        "attempted": window.attempted, "completed": window.completed,
        "phase": window.phase, "failed_substep": window.failed_substep,
        "substeps": tuple({
            "ordinal": item.ordinal, "accepted_worlds": item.accepted_worlds,
            "states": item.states, "diagnostics_complete": item.diagnostics_complete, "error": item.error,
        } for item in window.substeps),
        "action_receipt": result["action_receipt"],
    }
    return result


def continuation(output):
    """J D2, GRU, healthy update-boundary save/A/restore/B; no recovery bypass."""
    from phymeo.training.mujoco_tyx_checkpoint import MuJoCoTyxPpoSession
    from phymeo.training.rsl_mujoco_tyx import RslMuJoCoTyxVecEnv

    torch = require_cuda()
    source().tasks().seed_all(6511)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "FAILED_CONTROL_WINDOW_CONTINUATION", "seed": 6511, "case": "J", "decimation": 2}
    with (output / "J-d2-gru.jsonl").open("x", encoding="utf-8") as log:
        with make_environment("J", 2) as env:
            try:
                env.reset()
                session = MuJoCoTyxPpoSession(RslMuJoCoTyxVecEnv(env), steps=8, recurrent=True)
                warm = session.update()
                log.write(json_text({"event": "warm", "value": warm}) + "\n")
                nonzero = all(bool(torch.count_nonzero(value)) for value in session.training_state()["hidden"].values())
                if not nonzero:
                    raise AssertionError("actual actor/critic hidden state must be nonzero")
                checkpoint = output / "J-d2-gru-bundle"
                saved = session.save_checkpoint(checkpoint)
                a = session.update()
                train_a, physical_a = session.training_state(), env.runtime.snapshot()
                log.write(json_text({"event": "A", "value": a, "training": train_a, "physics": physical_a}) + "\n")
                env.reset((103, 7))
                next_a = (env.runtime.snapshot(), env.capture_training_state()["random_reset"])
                session.restore_checkpoint(checkpoint, expected_revision=env.runtime.checkpoint_revision)
                b = session.update()
                train_b, physical_b = session.training_state(), env.runtime.snapshot()
                log.write(json_text({"event": "B", "value": b, "training": train_b, "physics": physical_b}) + "\n")
                _assert_same(train_a, train_b, "complete learner/Adam/RNG/RNN/reset continuation differs")
                _assert_same(physical_a, physical_b, "physical continuation differs")
                _assert_same(tuple(map(_semantic_transition, a["transitions"])), tuple(map(_semantic_transition, b["transitions"])), "complete control-window continuation differs")
                _assert_same(a["loss"], b["loss"], "upstream loss differs")
                env.reset((7, 103))
                next_b = (env.runtime.snapshot(), env.capture_training_state()["random_reset"])
                _assert_same(next_a, next_b, "next subset sample after restore differs")
                masks = source()._mask_checks(warm) + source()._mask_checks(a) + source()._mask_checks(b)
                if masks == 0:
                    raise AssertionError("fixed recurrent trajectory did not exercise partial reset")
                with make_environment("J", 4) as wrong_env:
                    other = MuJoCoTyxPpoSession(RslMuJoCoTyxVecEnv(wrong_env), steps=8, recurrent=True)
                    before = wrong_env.runtime.checkpoint().to_bytes()
                    other_before = other.training_state()
                    try:
                        other.restore_checkpoint(checkpoint, expected_revision=wrong_env.runtime.checkpoint_revision)
                    except (ValueError, RuntimeError) as error:
                        if "binding" not in str(error):
                            raise
                    else:
                        raise AssertionError("cross-decimation restore was accepted")
                    if wrong_env.runtime.checkpoint().to_bytes() != before:
                        raise AssertionError("cross-D rejection changed physical state")
                    _assert_same(other.training_state(), other_before, "cross-D rejection changed learner/RNG")
                summary.update(
                    status="PASS_CONTROL_WINDOW_CONTINUATION", nonzero_hidden=bool(nonzero),
                    partial_done_masks=masks, cross_decimation_refused=True, next_random_reset_equal=True,
                    policy_transitions=32, physical_world_steps=64, save=saved,
                    binding=env.training_binding(),
                    exclusions=["monotonic runtime revision", "task epoch", "action publication revision/epoch", "resource counters"],
                )
            except BaseException as error:
                failure = env.last_failure
                log.write(json_text({"event": "failure", "error": repr(error), "needs_recovery": env.needs_recovery, "control_window": None if failure is None else failure.control_window}) + "\n")
                raise
            finally:
                (output / "J-d2-gru-summary.json").write_text(json_text(summary), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("J", "S", "gru"), required=True)
    parser.add_argument("--physics-steps", type=int, choices=(1, 2, 4))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.case == "gru":
        if args.physics_steps not in (None, 2):
            parser.error("GRU continuation fixes D=2")
        result = continuation(args.output)
    else:
        if args.physics_steps is None:
            parser.error("J/S require explicit --physics-steps")
        result = run_case(args.case, args.physics_steps, args.output)
    print(json_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fixed N7 CUDA control-window/random-reset scale qualification, not learning."""

import argparse
import importlib.util
import json
import sys
import traceback
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from phymeo.training import mujoco_tyx_evaluation as evidence

MATRIX = (("J", 8, 2, True), ("S", 128, 4, False), ("S", 256, 4, False))
_MODULES = {}


def load_example(filename, name):
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[name] = module
    return _MODULES[name]


def support():
    return load_example("compare_mujoco_tyx_observation_training.py", "n7_observation_support")


def tasks():
    return load_example("mujoco_tyx_learning_tasks.py", "n7_original_tasks")


def protocol(task, worlds):
    matches = [row for row in MATRIX if row[:2] == (task, worlds)]
    if type(worlds) is not int or len(matches) != 1:
        raise ValueError("fixed N7 matrix is J8/D2/GRU, S128/S256/D4/MLP")
    _, _, d, recurrent = matches[0]
    return dict(
        task=task,
        worlds=worlds,
        decimation=d,
        recurrent=recurrent,
        updates=3,
        warmup_updates=1,
        rollout_steps=8,
        seed=6511,
        namespace="N7",
        keys=tuple(7 + 13 * i for i in range(worlds)),
        horizons=tuple((3, 5, 7, 11)[i % 4] for i in range(worlds)),
        physical_worldsteps=24 * d * worlds,
        measured_worldsteps=16 * d * worlds,
        measured_policy_transitions=16 * worlds,
    )


def signature(value):
    return support().digest(value)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def owned(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: owned(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(owned(item) for item in value)
    if isinstance(value, list):
        return [owned(item) for item in value]
    # Public frozen receipts are already detached; do not duplicate large states.
    return value


def common_state(state):
    return {name: getattr(state, name) for name in support().STATE_FIELDS}


def common_action(receipt):
    result = {item.name: getattr(receipt, item.name) for item in fields(receipt)}
    result["binding"] = {
        key: value for key, value in receipt.binding.items() if key != "runtime_owner"
    }
    return result


def common_window(window):
    result = {item.name: getattr(window, item.name) for item in fields(window)}
    result["action_receipt"] = common_action(window.action_receipt)
    result["substeps"] = tuple(
        {
            **{item.name: getattr(sub, item.name) for item in fields(sub) if item.name != "states"},
            "states": None
            if sub.states is None
            else {key: common_state(state) for key, state in sub.states.items()},
        }
        for sub in window.substeps
    )
    return result


def common_transition(transition):
    result = {item.name: getattr(transition, item.name) for item in fields(transition)}
    result["states"] = {key: common_state(state) for key, state in transition.states.items()}
    result["action_receipt"] = common_action(transition.action_receipt)
    result["control_window"] = common_window(transition.control_window)
    for name in ("input_frame", "output_frame"):
        frame = getattr(transition, name)
        result[name] = dict(
            binding=support().stable_task_binding(frame.binding),
            revision=frame.revision,
            epoch=frame.epoch,
            ticks=frame.ticks,
            times=frame.times,
        )
    return result


def common_observed(attempts):
    rows = []
    for attempt in attempts:
        row = dict(attempt)
        row["transition"] = common_transition(row["transition"])
        observation, reward, done, extras = row["rsl_output"]
        extras = dict(extras)
        extras["control_window"] = common_window(extras["control_window"])
        row["rsl_output"] = (observation, reward, done, extras)
        rows.append(row)
    return tuple(rows)


@contextmanager
def observe_windows(adapter, env):
    """One public call; retain original requests and actual late-failure windows."""
    attempts, sentinel = [], object()
    previous, original = vars(adapter).get("step", sentinel), adapter.step

    def step(action):
        row = dict(ordinal=len(attempts), requested_action=owned(action), status="calling")
        attempts.append(row)
        try:
            result = original(action)
            row["transition"] = adapter.last_transition
            row["status"] = "accepted"
            row["rsl_output"] = owned(result)
            row["rsl_layout"] = tuple(
                dict(
                    type=type(value).__module__ + "." + type(value).__name__,
                    device=str(getattr(value, "device", None)),
                    batch_size=tuple(getattr(value, "batch_size", ())),
                )
                for value in result[:3]
            )
            row["reset"] = env.last_reset_receipt
            row["episode"] = env.capture_training_state()
            return result
        except BaseException as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error))
            for name in (
                "phase",
                "action_accepted",
                "state_accepted",
                "accepted_worlds",
                "needs_recovery",
                "action_receipt",
                "transition",
                "control_window",
            ):
                if hasattr(error, name):
                    row[name] = getattr(error, name)
            raise

    adapter.step = step
    try:
        yield attempts
    finally:
        if previous is sentinel:
            del adapter.step
        else:
            adapter.step = previous


@contextmanager
def make_environment(p, mode):
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState, MuJoCoRuntime
    from phymeo.backends.mujoco_task_workspace import MuJoCoTyxTaskSpec
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    from phymeo.training.mujoco_tyx_control_window import MuJoCoTyxControlWindowSpec
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxActionSpec, MuJoCoTyxVectorEnv
    from phymeo.training.mujoco_tyx_reset import MuJoCoTyxResetPlan

    original = tasks()
    item = original.task(p["task"])
    source = compile_mujoco_xml(item.xml, require_asset=False)
    width, grid = len(item.q_target), original.templates(p["task"], held_out=False)
    anchors = {key: MuJoCoInitialState(*grid[i % 8]) for i, key in enumerate(p["keys"])}
    qbounds = ((-0.1, 0.1),) * width if p["task"] == "J" else ((0.00065, 0.00085),)
    vbounds = ((-0.04, 0.04),) * width if p["task"] == "J" else ((-0.002, 0.002),)
    plan = MuJoCoTyxResetPlan(
        source,
        anchors,
        q_indices=tuple(range(width)),
        q_bounds=qbounds,
        v_indices=tuple(range(width)),
        v_bounds=vbounds,
        seed=p["seed"],
        namespace=p["namespace"],
    )
    spec = MuJoCoTyxTaskSpec(
        tuple(range(width)),
        tuple(range(width)),
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
    with ExitStack() as owners:
        runtime = owners.enter_context(
            MuJoCoRuntime(source, mode="cuda", world_keys=p["keys"], initial_states=anchors)
        )
        env = owners.enter_context(
            MuJoCoTyxVectorEnv(
                runtime,
                spec,
                action,
                h=item.h,
                horizons=p["horizons"],
                reset_plan=plan,
                observation_mode=mode,
                control_window=MuJoCoTyxControlWindowSpec(p["decimation"]),
            )
        )
        yield env, source


def check_window(transition, env, p):
    window = transition.control_window
    require(
        window.physics_steps == p["decimation"]
        and window.attempted == window.completed == p["decimation"]
        and window.phase == "complete"
        and window.failed_substep is None,
        "complete control window",
    )
    require(
        signature(window.action_receipt) == signature(transition.action_receipt), "one held action"
    )
    support_rows = 0
    previous = None
    dt = float(np.float32(env.h))
    for ordinal, sub in enumerate(window.substeps):
        require(
            sub.ordinal == ordinal
            and sub.diagnostics_complete
            and sub.error is None
            and dict(sub.accepted_worlds) == dict.fromkeys(env.world_keys, True),
            "complete substep",
        )
        require(sub.exit_revision == sub.entry_revision + 1, "one physical publication")
        if ordinal:
            require(
                sub.entry_revision == window.substeps[ordinal - 1].exit_revision,
                "substep revision chain",
            )
        require(tuple(sub.states) == tuple(env.world_keys), "complete ordered physical worlds")
        for slot, (key, state) in enumerate(sub.states.items()):
            receipt = state.last_step
            require(
                receipt.world == key
                and receipt.tick == state.tick
                and receipt.output_time == state.time,
                "substep world/tick/output time",
            )
            if previous is None:
                require(
                    receipt.input_time == transition.input_frame.times[slot]
                    and state.tick == transition.input_frame.ticks[slot] + 1,
                    "entry task/physics clock",
                )
            if previous is not None:
                require(
                    state.tick == previous[key].tick + 1
                    and receipt.input_time == previous[key].time,
                    "substep clock",
                )
                np.testing.assert_array_equal(receipt.input_q, previous[key].q)
                np.testing.assert_array_equal(receipt.input_v, previous[key].v)
            require(
                signature(receipt.stored_control)
                == signature(transition.action_receipt.controls[key]),
                "held native controls",
            )
            control = receipt.stored_control
            if p["task"] == "J":
                force = control.ctrl + control.qfrc_applied + control.xfrc_applied[1:3, 0]
                velocity = (1.01 * receipt.input_v + dt * force) / (1.01 + 0.1 * dt)
            else:
                support_force = np.zeros(6)
                for contact in receipt.contacts:
                    require(
                        contact.dimension == 1 and contact.normal_force >= 0, "original support law"
                    )
                    left, right = (
                        np.asarray(contact.wrench_on_body1_world),
                        np.asarray(contact.wrench_on_body2_world),
                    )
                    np.testing.assert_allclose(left[:3] + right[:3], 0.0, rtol=0, atol=2e-4)
                    if contact.body_ids[0] == 1:
                        support_force += left
                    if contact.body_ids[1] == 1:
                        support_force += right
                require(np.all(np.isfinite(support_force)), "finite published support")
                support_rows += int(support_force[2] > 0)
                external = float(
                    control.ctrl[0] + control.qfrc_applied[0] + control.xfrc_applied[1, 2]
                )
                velocity = receipt.input_v + dt * (external + support_force[2] - 2 * 9.81) / 2
            np.testing.assert_allclose(state.v, velocity, rtol=0, atol=3e-3)
            np.testing.assert_allclose(state.q, receipt.input_q + dt * velocity, rtol=0, atol=2e-4)
        previous = sub.states
    require(signature(transition.states) == signature(window.substeps[-1].states), "endpoint state")
    raw = np.asarray(
        [
            np.r_[s.q[list(env.task_spec.q_indices)], s.v[list(env.task_spec.v_indices)]]
            for s in transition.states.values()
        ],
        np.float32,
    )
    spec = env.task_spec
    np.testing.assert_allclose(
        evidence.host_array(transition.final_observation),
        raw * spec.scale + spec.bias,
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        evidence.host_array(transition.reward),
        -np.sum(spec.weight * (raw - spec.target) ** 2, axis=1),
        rtol=2e-5,
        atol=2e-6,
    )
    ended = np.any((raw < spec.lower) | (raw > spec.upper), axis=1)
    timeout = (transition.episode_length >= env.horizons) & ~ended
    np.testing.assert_array_equal(evidence.host_array(transition.terminated), ended)
    np.testing.assert_array_equal(evidence.host_array(transition.time_outs), timeout)
    np.testing.assert_array_equal(evidence.host_array(transition.final_mask), ended | timeout)
    return support_rows


def check_masks(record):
    partial = 0
    for mask in record["recurrent_masks"]:
        done = mask["done"]
        partial += int(bool(done.any()) and not bool(done.all()))
        for side in ("actor", "critic"):
            before, after = mask["before"][side], mask["after"][side]
            require(
                bool((before[:, ~done] == after[:, ~done]).all()), "non-done GRU rows preserved"
            )
            require(not bool(after[:, done].count_nonzero()), "done GRU rows cleared")
    return partial


def mode_summary(value):
    # Never return an env, Tensor, native model or rollout into the next factory.
    return json.loads(json.dumps(value, allow_nan=False))


def run_mode(p, mode, target):
    import mujoco as mj

    from phymeo.backends import _mujoco_tyx_runtime
    from phymeo.training.mujoco_tyx_checkpoint import MuJoCoTyxPpoSession
    from phymeo.training.rsl_mujoco_tyx import RslMuJoCoTyxVecEnv

    original, helpers = tasks(), support()
    original.require_cuda()
    original.seed_all(p["seed"])
    require(mj.__version__ == "3.3.7", "fixed native compiler")
    writer = evidence.EvidenceWriter(target / (mode + ".jsonl.gz"))
    memory = helpers.MemoryObservation(target, p["worlds"])
    # Observational capacity only, not a physical/resource-budget change. At most
    # 24 all-world resets, each with candidate + detached observation construction.
    memory.maximum_constructors = 4 * p["worlds"] * (24 + 2) + 128
    memory.maximum_copies = 2 * (24 * p["decimation"] + 24 + 2) * p["worlds"] + 128
    memory.maximum_events = 8 * 24 + 128
    data_type = mj.MjData
    signatures, timings = [], []
    partial_masks, supported, nonzero_hidden = 0, 0, False
    measured_start, write_start, current_attempts = None, None, []
    before_cold = perf_counter()
    summary = {"status": "FAILED_N7_MODE", "mode": mode, "protocol": p}
    forbidden_calls = []

    def native_snapshot():
        value = memory.snapshot()
        value["step_index_semantics"] = (
            "policy-window ordinal; physics_steps has D records per window"
        )
        value["physics_steps_per_window"] = p["decimation"]
        return value

    def forbidden(*args, **kwargs):
        forbidden_calls.append("CPU integration attempted")
        raise AssertionError("CPU step/step1/step2 cannot implement TYX physics")

    try:
        with (
            memory.install(_mujoco_tyx_runtime, data_type, native_module=mj),
            ExitStack() as guards,
        ):
            for name in ("mj_step", "mj_step1", "mj_step2"):
                guards.enter_context(patch.object(mj, name, forbidden))
            memory.begin_mode(mode)
            try:
                with ExitStack() as owners:
                    env, source = owners.enter_context(make_environment(p, mode))
                    owners.enter_context(memory.observe_runtime(env.runtime, data_type))
                    adapter = RslMuJoCoTyxVecEnv(env)
                    owners.enter_context(memory.observe_steps(adapter))
                    session = MuJoCoTyxPpoSession(adapter, steps=8, recurrent=p["recurrent"])
                    initial = session.training_state()
                    stable_binding = helpers.stable_environment_binding(env.binding)
                    initial_hash, binding_hash = signature(initial), signature(stable_binding)
                    writer.write(
                        "initial",
                        dict(
                            protocol=p,
                            source=source.source,
                            binding=env.binding,
                            adapter_config=adapter.cfg,
                            learner_config=session.config,
                            training=initial,
                            frame=env.current_frame,
                            observation=env.get_observations(),
                            states=env.runtime.integration_snapshot(),
                        ),
                    )
                    observation_hash = signature(env.get_observations())
                    memory.finish_construction()
                    memory.event("construct-complete")
                    cold_seconds = perf_counter() - before_cold
                    # Release the copied initial state before measured rollouts.
                    del initial
                    for iteration in range(3):
                        if iteration == 1:
                            measured_start, write_start = perf_counter(), writer.write_seconds
                        with observe_windows(adapter, env) as current_attempts:
                            record, timing = evidence.measure_update(session, observe=False)
                        writer.write("window_requests_and_rsl_outputs", current_attempts)
                        checked = evidence.validate_update(
                            record,
                            env,
                            expected_iteration=iteration,
                            cost=original.task(p["task"]).cost(),
                            writer=writer,
                        )
                        require(
                            len(record["transitions"]) == len(current_attempts) == 8,
                            "eight complete policy windows",
                        )
                        for ordinal, (transition, attempt) in enumerate(
                            zip(record["transitions"], current_attempts, strict=True)
                        ):
                            require(
                                attempt["ordinal"] == ordinal
                                and attempt["status"] == "accepted"
                                and attempt["transition"] is transition,
                                "complete public prefix binding",
                            )
                            require(
                                signature(attempt["requested_action"])
                                == signature(transition.action),
                                "requested policy action",
                            )
                            supported += check_window(transition, env, p)
                            done_keys = tuple(
                                key
                                for key, done in zip(
                                    env.world_keys,
                                    evidence.host_array(transition.final_mask),
                                    strict=True,
                                )
                                if done
                            )
                            if done_keys:
                                require(
                                    tuple(attempt["reset"].world_keys) == done_keys,
                                    "actual reset owners",
                                )
                                require(
                                    not attempt["reset"].cleanup_errors,
                                    "reset owner cleanup failed",
                                )
                                require(
                                    attempt["episode"]["random_reset"]["last_sample"]["world_keys"]
                                    == done_keys,
                                    "actual reset ledger",
                                )
                        partial_masks += check_masks(record)
                        training = session.training_state()
                        if p["recurrent"]:
                            nonzero_hidden = nonzero_hidden or all(
                                bool(value.count_nonzero()) for value in training["hidden"].values()
                            )
                        common = dict(record)
                        common["transitions"] = tuple(
                            common_transition(tr) for tr in record["transitions"]
                        )
                        signatures.append(
                            dict(
                                iteration=iteration,
                                update=signature(common),
                                observed=signature(common_observed(current_attempts)),
                                training=signature(training),
                            )
                        )
                        writer.write("training_state", training)
                        writer.write("measurement", dict(timing=timing, independent_check=checked))
                        if iteration:
                            timings.append(timing)
                        # Avoid retaining the previous large update while collecting another.
                        del training, common, record, current_attempts
                        current_attempts = []
                    require(session.next_iteration == 3, "three actual upstream updates")
                    counters = evidence.plain_counters(env.counters)
                    require(
                        counters["transitions"] == 24
                        and counters["physical_steps"] == 24 * p["decimation"],
                        "policy and physical counters",
                    )
                    require(counters["resets"] > 0, "real random subset resets")
                    require(
                        env.runtime.counters["device_control_submissions"] == 24,
                        "one native action per window",
                    )
                    require(
                        len(memory.physics_steps) == 24 * p["decimation"],
                        "all physical steps observed",
                    )
                    if p["task"] == "S":
                        require(supported > 0, "actual loaded support observed")
                    if p["recurrent"]:
                        require(
                            nonzero_hidden and partial_masks > 0,
                            "nonzero GRU and actual partial done mask",
                        )
                    require(not forbidden_calls, "no CPU integration fallback")
                    require(
                        memory.constructor_failures == memory.copy_failures == 0,
                        "native allocation/copy failure",
                    )
                    require(
                        env.runtime.counters["failed_batches"] == 0, "no rejected physical batch"
                    )
                    memory.event("mode-close-enter")
            finally:
                memory.finish_construction(failed=True)
                memory.event("mode-close-after-owner-attempts")
        writer.close()
        wall = perf_counter() - measured_start
        summary.update(
            status="PASS_N7_MODE",
            initial_training_hash=initial_hash,
            initial_observation_hash=observation_hash,
            stable_binding_hash=binding_hash,
            cold_environment_learner_s=cold_seconds,
            signatures=signatures,
            measured_updates=timings,
            counter_end=counters,
            physical_worldsteps=p["physical_worldsteps"],
            measured_worldsteps=p["measured_worldsteps"],
            partial_gru_masks=partial_masks,
            nonzero_gru_hidden=nonzero_hidden,
            supported_world_substeps=supported,
            compute_and_mandatory_diagnostics_s=sum(
                x["compute_and_mandatory_diagnostics_s"] for x in timings
            ),
            loop_to_trace_close_s=wall,
            evidence_write_s=writer.write_seconds - write_start,
            policy_transitions_per_s_with_trace=p["measured_policy_transitions"] / wall,
            physical_worldsteps_per_s_with_trace=p["measured_worldsteps"] / wall,
            native_observation=native_snapshot(),
            cpu_integration_calls=len(forbidden_calls),
            timing_scope="Includes observers, diagnostics, evidence and update hashes; final file hash/summary excluded",
            performance_eligible=False,
            exclusive_host_or_unobserved_sync="UNKNOWN",
            evidence=writer.receipt(),
        )
        # The finally block writes this exclusive summary exactly once.
        return mode_summary(summary)
    except BaseException as error:
        summary.update(
            status="FAILED_N7_MODE",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        try:
            writer.write("failed_window_prefix", current_attempts)
            if isinstance(error, evidence.UpdateCaptureError):
                writer.write("upstream_failed_attempts", error.attempts)
                writer.write("upstream_completed_record", error.completed_record)
            writer.write("failure", summary)
        except BaseException:
            summary["failure_evidence_error"] = traceback.format_exc()
        raise
    finally:
        # Do not query closed owners; attempt all finalizers without masking a
        # primary physics/learner failure with a later diagnostic exception.
        primary, failures = sys.exc_info()[1], []
        for stage, operation in (
            ("native-observation", lambda: summary.update(native_observation=native_snapshot())),
            ("evidence-close", writer.close),
            (
                "summary-write",
                lambda: evidence.json_write(target / (mode + "-summary.json"), summary),
            ),
        ):
            try:
                operation()
            except BaseException as error:
                summary["status"] = "FAILED_N7_MODE"
                summary.setdefault("finalization_errors", []).append(
                    dict(stage=stage, error=repr(error))
                )
                failures.append(error)
        if failures:
            raise BaseExceptionGroup(
                "N7 primary/finalization failures",
                ([primary] if primary is not None else []) + failures,
            )


def run_pair(task, worlds, output):
    p = protocol(task, worlds)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="FAILED_N7_PAIRED_WINDOW_SCALE",
        protocol=p,
        modes={},
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    try:
        for mode in ("full", "integration"):
            result["modes"][mode] = run_mode(p, mode, output)
        full, compact = result["modes"]["full"], result["modes"]["integration"]
        for name in (
            "initial_training_hash",
            "initial_observation_hash",
            "stable_binding_hash",
            "signatures",
        ):
            require(full[name] == compact[name], "full/integration common comparison: " + name)
        result.update(
            status="PASS_N7_PAIRED_WINDOW_SCALE",
            common_physics_training_identical=True,
            exclusions=(
                "opaque runtime/workspace owners",
                "full-only endpoint geometry/forces/sensors/energy fields",
            ),
            measured_pairs=2,
            learning_claim=False,
            newton_speed_claim=False,
        )
        return mode_summary(result)
    except BaseException as error:
        result.update(
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        evidence.json_write(output / "summary.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("J", "S"))
    parser.add_argument("--worlds", type=int)
    parser.add_argument(
        "--all", action="store_true", help="fixed order J8, S128, S256; stop at first failure"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.all:
        if args.task is not None or args.worlds is not None:
            parser.error("--all cannot be combined with a cell")
        args.output.mkdir(parents=True, exist_ok=False)
        rows = []
        for task, worlds, _, _ in MATRIX:
            rows.append(run_pair(task, worlds, args.output / f"{task}-w{worlds}"))
        evidence.json_write(
            args.output / "summary.json", dict(status="PASS_N7_FIXED_THREE", cells=rows)
        )
    else:
        if args.task is None or args.worlds is None:
            parser.error("select one fixed cell or --all")
        result = run_pair(args.task, args.worlds, args.output)
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

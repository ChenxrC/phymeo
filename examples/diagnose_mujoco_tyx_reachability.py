"""One fixed J reachability witness; no policy training or controller search.

The original eight held-out worlds run exactly 256 public CUDA VecEnv steps.
Historical controls are metadata-bound comparisons, never rerun or reclassified.
"""

import argparse
import importlib.util
import json
import math
import sys
import traceback
from hashlib import sha256
from pathlib import Path

import numpy as np

from phymeo.training import mujoco_tyx_evaluation as evidence
from phymeo.training.learning_diagnostics import require, strict_json

STEPS, SWITCH, H = 256, 128, .002
WORLD_KEYS = tuple(5 + 36 * i for i in range(8))
Q_ATOL, V_ATOL = 2e-4, 3e-3
MASS, DAMPING = 1.01, .1
XML_SHA256 = "7883e8b64f549b7cd4e83f6eb59a68825761b9caeb27894d5f13f79178ea530f"
N4D_SHA256 = "107a2fbfd55b889b2e174f4900db9ab767547575e88450342febe58089e5b193"
MAX_METADATA_BYTES = 4 * 1024**2


def _example(filename):
    name = "n4e_reused_" + Path(filename).stem
    if name not in sys.modules:
        path = Path(__file__).with_name(filename)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


def learning_api():
    return _example("evaluate_mujoco_tyx_learning.py")


def protocol_binding():
    result = learning_api().protocol_binding("J")
    require(result["xml_sha256"] == XML_SHA256, "frozen J XML")
    require(
        result["h"] == H and result["horizon"] == STEPS
        and result["learning_worlds"] == 8 and result["action_scale"] == 2
        and result["action_squash"] == "tanh" and result["q_target"] == [0., 0.]
        and result["q_lower"] == [-1., -1.] and result["q_upper"] == [1., 1.]
        and result["v_bound"] == 8 and result["seeds"] == [14011, 14012, 14013],
        "frozen J protocol",
    )
    return result


def read_metadata(path, expected=None):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= MAX_METADATA_BYTES, "bounded metadata")
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    require(expected is None or digest == expected, "metadata hash")
    return strict_json(raw), digest


def load_history(evidence_root, ablation_root, protocol):
    # This is the existing bounded metadata reader, not its learning entry.
    old_api = _example("evaluate_mujoco_tyx_initialization_ablation.py")
    old_binding, originals = old_api.load_originals(evidence_root)
    directory = Path(ablation_root).resolve()
    variant, digest = read_metadata(directory / "summary.json", N4D_SHA256)
    seeds = [14011, 14012, 14013]
    require(
        variant["schema"] == "n4d-j-ablation-comparison-v1"
        and variant["classification"] == "FIXED_BUDGET_NOT_LEARNED"
        and variant["original_classification"] == "FIXED_BUDGET_NOT_LEARNED"
        and [row["seed"] for row in variant["rows"]] == seeds,
        "complete fixed N4d identities/outcome",
    )
    aggregate = variant["variant_three_seed"]
    require(
        aggregate["task"] == "J" and aggregate["seeds"] == seeds
        and aggregate["protocol_binding"] == protocol
        and aggregate["outcome"]["classification"] == "FIXED_BUDGET_NOT_LEARNED"
        and set(aggregate["source_bindings"]) == set(map(str, seeds)),
        "N4d fixed protocol/bindings",
    )
    require(
        variant["original_bindings"]["anchor_sha256"] == old_binding["anchor_sha256"],
        "N4d original historical anchor",
    )
    rows = []
    variant_zero = None
    for row, seed in zip(variant["rows"], seeds, strict=True):
        original, bound = originals[seed], aggregate["source_bindings"][str(seed)]
        current_directory = directory / ("J-" + str(seed))
        require(Path(bound["directory"]).resolve() == current_directory, "N4d run directory")
        require(Path(row["directory"]).resolve() == current_directory, "N4d row directory")
        saved, saved_hash = read_metadata(current_directory / "summary.json", bound["summary_sha256"])
        publication, _ = read_metadata(current_directory / "publication.json")
        manifest, manifest_hash = read_metadata(current_directory / "manifest.json")
        require(
            saved["task"] == "J" and saved["seed"] == seed
            and saved["status"] == "COMPLETE_N4A_SEED" and saved["updates"] == 128
            and saved["worlds"] == 8 and saved["accepted_world_steps"] == 32768
            and saved["evidence"] == bound["evidence"]
            and publication["summary_sha256"] == saved_hash
            and row["status"] == "COMPLETE", "complete N4d run/publication",
        )
        for key in ("task", "seed", "protocol_binding", "ppo_config", "source_files", "dependencies"):
            require(saved[key] == manifest[key], "N4d manifest binding: " + key)
        require(original["protocol_binding"] == saved["protocol_binding"] == protocol, "historical task")
        require(
            row["baseline_cost"] == saved["baseline"]["cost"]
            and row["final_cost"] == saved["learned"]["cost"]
            and row["original_baseline_cost"] == original["baseline"]["cost"]
            and row["original_final_cost"] == original["learned"]["cost"],
            "historical cost copies",
        )
        if seed == 14011:
            variant_zero = saved["zero_action_diagnostic"]
        for phase in ("baseline", "learned"):
            require(saved[phase]["world_steps"] == 2048, "complete historical held-out window")
        rows.append({
            "seed": seed, "original_baseline": original["baseline"]["cost"],
            "original_final": original["learned"]["cost"],
            "centered_baseline": saved["baseline"]["cost"], "centered_final": saved["learned"]["cost"],
            "original_source_files": original["source_files"], "variant_source_files": saved["source_files"],
            "original_dependencies": original["dependencies"], "variant_dependencies": saved["dependencies"],
            "variant_summary_sha256": saved_hash, "variant_manifest_sha256": manifest_hash,
            "variant_evidence": saved["evidence"],
        })
    zero = aggregate["zero_action_diagnostic"]
    require(zero["status"] == "COMPLETE_DIAGNOSTIC" and zero["world_steps"] == 2048, "historical zero")
    require(zero == variant_zero, "N4d zero diagnostic publication")
    return {
        "scope": "historical metadata only; no old trace read or contemporaneous control",
        "original_binding": old_binding, "variant_summary_sha256": digest,
        "variant_summary_path": str(directory / "summary.json"), "rows": rows,
        "original_zero_cost": originals[14011]["zero_action_diagnostic"]["cost"],
        "centered_zero_cost": zero["cost"], "original_learning_classification": "FIXED_BUDGET_NOT_LEARNED",
        "variant_learning_classification": "FIXED_BUDGET_NOT_LEARNED",
    }


class ReferenceInfeasible(ValueError):
    pass


def _frozen(value, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def advance_reference(q, v, control, h):
    """Native-parameter FP64 Euler identity, not a copy of the device kernel."""
    velocity = (MASS * np.asarray(v, np.float64) + h * np.asarray(control, np.float64)) / (MASS + h * DAMPING)
    return np.asarray(q, np.float64) + h * velocity, velocity


def _reference_path(q, v, force, h):
    positions, velocities = [np.asarray(q, np.float64).copy()], [np.asarray(v, np.float64).copy()]
    for row in force:
        q, v = advance_reference(q, v, row, h)
        positions.append(q.copy())
        velocities.append(v.copy())
    return np.asarray(positions), np.asarray(velocities)


def make_schedule(q, v):
    """The only admitted schedule: eight two-axis starts and a 128/128 split."""
    q, v = np.asarray(q), np.asarray(v)
    if q.shape != (8, 2) or v.shape != (8, 2) or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ReferenceInfeasible("finite eight-world/two-axis initial states required")
    if np.any(np.abs(q) >= 1) or np.any(np.abs(v) >= 8):
        raise ReferenceInfeasible("initial state outside original bounds")
    # A caller cannot substitute unrepresented FP64 initial coordinates.
    if not np.array_equal(q, q.astype(np.float32).astype(q.dtype)) or not np.array_equal(v, v.astype(np.float32).astype(v.dtype)):
        raise ReferenceInfeasible("initial states must be actual stored FP32 values")
    q, v, h = q.astype(np.float64), v.astype(np.float64), float(np.float32(H))
    a, b = MASS / (MASS + h * DAMPING), h / (MASS + h * DAMPING)
    matrix, column = np.asarray([[1, h*a], [0, a]]), np.asarray([h*b, b])
    power, summed = np.eye(2), np.zeros(2)
    for _ in range(SWITCH):
        power, summed = matrix @ power, matrix @ summed + column
    system = np.column_stack((power @ summed, summed))
    entry = np.stack((q, v), axis=-1).reshape(16, 2)
    rhs = -(power @ power @ entry.T)
    try:
        pulses = np.linalg.solve(system, rhs).T.reshape(8, 2, 2)
    except np.linalg.LinAlgError as error:
        raise ReferenceInfeasible("singular fixed two-pulse system") from error
    force = np.concatenate((np.broadcast_to(pulses[:, :, 0], (SWITCH, 8, 2)),
                            np.broadcast_to(pulses[:, :, 1], (SWITCH, 8, 2))))
    if not np.isfinite(force).all() or np.any(np.abs(force) >= 2):
        raise ReferenceInfeasible("fixed two-pulse force is outside strict (-2,2); no clipping")
    raw = np.arctanh(force / 2).astype(np.float32)
    if not np.isfinite(raw).all():
        raise ReferenceInfeasible("inverse tanh is not finite FP32")
    mapped = 2 * np.tanh(raw.astype(np.float64))
    ideal_q, ideal_v = _reference_path(q, v, force, h)
    mapped_q, mapped_v = _reference_path(q, v, mapped, h)
    for positions, velocities in ((ideal_q, ideal_v), (mapped_q, mapped_v)):
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all() or np.any(np.abs(positions) >= 1) or np.any(np.abs(velocities) >= 8):
            raise ReferenceInfeasible("complete fixed reference exceeds original state bounds")
    return {
        "schema": "n4e-j-two-pulse-reference-v1", "steps": STEPS, "switch": SWITCH,
        "h_nominal": H, "h_stored": h, "mass_native": MASS, "damping_native": DAMPING,
        "initial_q": _frozen(q), "initial_v": _frozen(v), "force": _frozen(force),
        "raw_action": _frozen(raw, np.float32), "mapped_host_diagnostic_force": _frozen(mapped),
        "ideal_reference_q": _frozen(ideal_q), "ideal_reference_v": _frozen(ideal_v),
        "mapped_reference_q": _frozen(mapped_q), "mapped_reference_v": _frozen(mapped_v),
        "system": _frozen(system), "rhs": _frozen(rhs),
        "solve_max_abs_residual": float(np.max(np.abs(system @ pulses.reshape(16, 2).T - rhs))),
        "system_condition_2": float(np.linalg.cond(system)),
        "planned_force_margin": float(2 - np.max(np.abs(force))),
        "host_tanh_scope": "preflight only; actual CUDA stored controls drive the independent runtime recurrence",
    }


def _array(value, shape, label):
    result = evidence.host_array(value)
    require(result.shape == shape and np.isfinite(result).all(), label + " shape/finite")
    return result


def validate_transition(transition, frame, raw, tick, previous_q, previous_v):
    from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxStepReceipt
    from phymeo.backends.mujoco_device_controls import MuJoCoTyxDeviceControlReceipt
    from phymeo.backends.mujoco_runtime import MuJoCoState
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxTransition

    require(type(transition) is MuJoCoTyxTransition, "actual public transition schema")
    evidence.check_window_transition(transition, WORLD_KEYS, tick, STEPS)
    require(
        transition.sequence == tick and tuple(transition.world_keys) == WORLD_KEYS
        and transition.input_frame is frame, "transition/input-frame identity",
    )
    require(np.array_equal(_array(transition.action, (8, 2), "action"), raw), "requested action binding")
    require(np.all(_array(transition.time_outs, (8,), "timeouts") == (tick == STEPS)), "complete timeout sequence")
    require(np.all(_array(transition.final_mask, (8,), "final mask") == (tick == STEPS)), "complete final mask")
    require(np.all(_array(transition.episode_length, (8,), "length") == tick)
            and not _array(transition.episode_id, (8,), "episode id").any(), "episode identity")
    require(len(transition.native_channels) == 3, "three native channels")
    ctrl = _array(transition.native_channels[0], (8, 2), "ctrl")
    qfrc = _array(transition.native_channels[1], (8, 2), "qfrc")
    xfrc = _array(transition.native_channels[2], (8, 3, 6), "xfrc")
    require(ctrl.dtype == np.float32 and qfrc.dtype == np.float32 and xfrc.dtype == np.float32,
            "actual FP32 native channels")
    require(not qfrc.any() and not xfrc.any() and np.all(np.abs(ctrl) < 2), "original force domain")
    action_receipt = transition.action_receipt
    require(type(action_receipt) is MuJoCoTyxDeviceControlReceipt
            and action_receipt.source_revision == frame.revision
            and action_receipt.revision == frame.revision + 1
            and set(action_receipt.controls) == set(WORLD_KEYS), "action transaction binding")
    require(len(frame.ticks) == len(frame.times) == 8, "complete input clocks")
    q, v = [], []
    for row, key in enumerate(WORLD_KEYS):
        state = transition.states[key]
        require(type(state) is MuJoCoState and state.world == key and state.tick == tick, "state world/tick")
        receipt = state.last_step
        require(type(receipt) is MuJoCoTyxStepReceipt and receipt.backend == "mujoco-tyx-cuda"
                and receipt.world == key and receipt.tick == tick, "actual step world/tick/backend")
        require(frame.ticks[row] == tick-1 and receipt.input_time == frame.times[row]
                and receipt.output_time == state.time and receipt.contact_time == receipt.input_time
                and math.isfinite(state.time) and 0 <= receipt.input_time < state.time
                and receipt.h == H, "actual step clock binding")
        require(np.array_equal(receipt.input_q, previous_q[row])
                and np.array_equal(receipt.input_v, previous_v[row]), "complete step entry state")
        for control in (receipt.control, receipt.stored_control, state.held_control, action_receipt.controls[key]):
            require(np.array_equal(_array(control.ctrl, (2,), "held ctrl"), ctrl[row])
                    and np.array_equal(_array(control.qfrc_applied, (2,), "held qfrc"), qfrc[row])
                    and np.array_equal(_array(control.xfrc_applied, (3, 6), "held xfrc"), xfrc[row]),
                    "stored-control/native-channel binding")
        require(np.array_equal(state.ctrl, ctrl[row]) and np.array_equal(state.qfrc_applied, qfrc[row])
                and np.array_equal(state.xfrc_applied, xfrc[row]), "state native channels")
        require(not receipt.contacts and not receipt.constraints and not state.contacts,
                "original noncontact J domain")
        require(not np.asarray(receipt.qfrc_constraint).any()
                and np.array_equal(receipt.qfrc_actuator, ctrl[row]), "actual motor force mapping")
        q.append(_array(state.q, (2,), "q").astype(np.float64))
        v.append(_array(state.v, (2,), "v").astype(np.float64))
    q, v = np.asarray(q), np.asarray(v)
    require(np.all(np.abs(q) < 1) and np.all(np.abs(v) < 8), "original full-window state bounds")
    return q, v, ctrl.astype(np.float64)


class WindowFailure(RuntimeError):
    def __init__(self, detail):
        super().__init__(detail["error_type"] + ": " + detail["message"])
        self.detail = detail


def submit_raw(env, raw):
    import torch

    with torch.inference_mode():
        action = torch.from_numpy(raw.copy()).to(device=env.device, dtype=torch.float32)
        return env.step(action, world_keys=env.world_keys, expected_frame=env.current_frame,
                        producer_stream=torch.cuda.current_stream(env.device))


def run_window(env, schedule, writer, *, submit=submit_raw):
    """One full public window. The injection seam is only for synthetic HOST tests."""
    tick, validated, saved, received, phase = 0, 0, 0, 0, "entry"
    transition, requested = None, None
    try:
        require(tuple(env.world_keys) == WORLD_KEYS and env.num_envs == 8 and env.h == H, "fixed live window")
        require(schedule["steps"] == STEPS and schedule["switch"] == SWITCH
                and schedule["h_stored"] == float(np.float32(H)), "fixed schedule identity")
        cost = learning_api().tasks().task("J").cost()
        previous_q, previous_v = schedule["initial_q"].copy(), schedule["initial_v"].copy()
        ref_q, ref_v = previous_q.copy(), previous_v.copy()
        total = np.zeros(8, np.float64)
        max_q, max_v = np.zeros((8, 2)), np.zeros((8, 2))
        reward_error, force_margin = 0., 2.
        for tick in range(1, STEPS+1):
            transition = None
            raw, frame = schedule["raw_action"][tick-1], env.current_frame
            requested = {"tick": tick, "raw_action": raw, "input_frame": frame,
                         "planned_force": schedule["force"][tick-1]}
            phase = "requested-action-record"
            writer.write("requested_action", requested)
            phase = "public-step"
            transition = submit(env, raw)
            received += 1
            phase = "accepted-transition-record"
            writer.write("transition", transition)
            saved += 1
            phase = "transition-validation"
            q, v, actual_force = validate_transition(transition, frame, raw, tick, previous_q, previous_v)
            ref_q, ref_v = advance_reference(ref_q, ref_v, actual_force, schedule["h_stored"])
            q_error, v_error = np.abs(q-ref_q), np.abs(v-ref_v)
            values = evidence.native_cost(transition.states, WORLD_KEYS, cost)
            reward = _array(transition.reward, (8,), "reward").astype(np.float64)
            phase = "independent-evidence"
            writer.write("independent", {
                "tick": tick, "actual_stored_force": actual_force,
                "reference_q": ref_q, "reference_v": ref_v,
                "q_absolute_error": q_error, "v_absolute_error": v_error,
                "cost": values, "reward_absolute_error": np.abs(reward+values),
            })
            phase = "independent-gates"
            require(np.all(q_error <= Q_ATOL) and np.all(v_error <= V_ATOL), "actual/reference disagreement")
            require(np.allclose(reward, -values, rtol=2e-5, atol=2e-6), "original reward/native cost disagreement")
            max_q, max_v = np.maximum(max_q, q_error), np.maximum(max_v, v_error)
            reward_error = max(reward_error, float(np.max(np.abs(reward+values))))
            force_margin = min(force_margin, float(2-np.max(np.abs(actual_force))))
            total += values
            previous_q, previous_v = q, v
            validated += 1
        phase = "terminal-gates"
        require(validated == STEPS and received == saved == STEPS, "complete fixed window")
        require(np.all(np.abs(q) <= Q_ATOL) and np.all(np.abs(v) <= V_ATOL), "terminal target disagreement")
        require(np.all(env.episode_length == 0) and np.all(env.episode_id == 1), "all eight terminal resets")
        return {
            "classification": "PASS_FIXED_REFERENCE_DIAGNOSTIC", "steps": STEPS,
            "world_steps": STEPS*8, "cost": float(np.mean(total/STEPS)),
            "per_world_cost": (total/STEPS).tolist(), "terminal_q": q.tolist(), "terminal_v": v.tolist(),
            "reference_terminal_q": ref_q.tolist(), "reference_terminal_v": ref_v.tolist(),
            "max_q_absolute_error": max_q.tolist(), "max_v_absolute_error": max_v.tolist(),
            "reward_max_absolute_error": reward_error, "actual_force_margin": force_margin,
            "terminal_origin": "transition256.states before the original automatic subset reset",
            "scope": "one bounded diagnostic controller; not a learned policy or proof for R",
            "learning_classification": "UNCHANGED_FIXED_BUDGET_NOT_LEARNED",
            "newton_performance_claim": False,
        }
    except Exception as error:
        detail = {
            "classification": "FAILED_REFERENCE_DIAGNOSTIC", "phase": phase, "tick": tick,
            "error_type": type(error).__name__, "message": str(error), "cost": None,
            "transitions_received": received, "transitions_saved": saved, "validated_steps": validated,
            "last_request": requested,
            "unwritten_transition": transition if received > saved else None,
        }
        for name in ("phase", "action_accepted", "state_accepted", "reset_accepted", "accepted_worlds", "needs_recovery", "action_receipt"):
            if hasattr(error, name):
                detail["error_" + name] = getattr(error, name)
        raise WindowFailure(detail) from error


def native_model_metadata(env, protocol):
    """Read a fresh public compiled model; never advance CPU dynamics."""
    import mujoco as mj

    compilation = env.runtime.compilation
    source = compilation.source
    require(source.primary_sha256 == protocol["xml_sha256"] == XML_SHA256, "compiled source binding")
    require(source.mujoco_version == "3.3.7", "fixed native compiler version")
    model = compilation.fresh_native_model()
    require((model.nq, model.nv, model.nu, model.nbody, model.ngeom) == (2, 2, 2, 3, 2), "original native layout")
    require(float(model.opt.timestep) == H and model.opt.integrator == mj.mjtIntegrator.mjINT_EULER
            and not np.asarray(model.opt.gravity).any(), "original Euler/gravity")
    require(np.array_equal(model.jnt_type, [mj.mjtJoint.mjJNT_SLIDE]*2)
            and np.array_equal(model.jnt_bodyid, [1, 2])
            and np.array_equal(model.body_parentid[1:], [0, 0])
            and np.array_equal(model.jnt_qposadr, [0, 1])
            and np.array_equal(model.jnt_dofadr, [0, 1])
            and np.array_equal(model.jnt_axis, [[1, 0, 0]]*2), "independent native slide axes")
    require(np.array_equal(model.body_mass[1:], [1., 1.])
            and np.array_equal(model.dof_armature, [.01, .01])
            and np.array_equal(model.dof_damping, [.1, .1])
            and not np.asarray(model.jnt_stiffness).any()
            and not np.asarray(model.jnt_limited).any()
            and not np.asarray(model.geom_contype).any()
            and not np.asarray(model.geom_conaffinity).any(), "original scalar dynamics/contact domain")
    require(np.array_equal(model.actuator_trnid[:, 0], [0, 1])
            and np.array_equal(model.actuator_gear[:, 0], [1., 1.])
            and not np.asarray(model.actuator_gear[:, 1:]).any()
            and np.all(model.actuator_ctrllimited) and np.all(model.actuator_forcelimited)
            and np.array_equal(model.actuator_ctrlrange, [[-2., 2.]]*2)
            and np.array_equal(model.actuator_forcerange, [[-2., 2.]]*2), "original native motor range")
    fields = ("body_mass", "body_parentid", "dof_armature", "dof_damping", "jnt_type", "jnt_bodyid",
              "jnt_qposadr", "jnt_dofadr", "jnt_axis", "actuator_gear", "actuator_trnid",
              "actuator_ctrlrange", "actuator_forcerange", "actuator_gainprm", "actuator_biasprm")
    values = {name: np.array(getattr(model, name), copy=True) for name in fields}
    return {
        "source": source, "fields": values, "nominal_h": H, "stored_h": float(np.float32(H)),
        "independent_effective_mass": model.body_mass[1:].copy()+model.dof_armature.copy(),
        "fp32_parameter_diagnostics": {name: np.asarray(values[name], np.float32)
                                       for name in ("body_mass", "dof_armature", "dof_damping")},
        "scope": "public native metadata; no mj_step/mj_forward reference trajectory",
        "arithmetic_boundary": "kind0 may use precomputed FP32 inverse/Euler operators; scalar FP64 recurrence is an engineering oracle, not identical operations",
    }


def validate_entry(env, protocol):
    from phymeo.backends.mujoco_runtime import MuJoCoState

    require(tuple(env.world_keys) == WORLD_KEYS and env.h == H and env.num_envs == 8, "live J world order")
    require(np.array_equal(env.horizons, [STEPS]*8) and env.observation_mode == "full", "original horizon/observation mode")
    require(env.runtime.capabilities["implementation"] == "tyx"
            and env.runtime.capabilities["mode"] == "cuda", "real CUDA runtime required")
    require(env.action_spec.ctrl_indices == (0, 1) and not env.action_spec.qfrc_indices
            and not env.action_spec.xfrc_indices and np.array_equal(env.action_spec.scale, [2, 2])
            and not env.action_spec.bias.any(), "original public action mapping")
    require(env.task_spec.q_indices == (0, 1) and env.task_spec.v_indices == (0, 1)
            and np.array_equal(env.task_spec.scale, [5, 5, 1, 1]) and not env.task_spec.bias.any()
            and not env.task_spec.target.any()
            and np.array_equal(env.task_spec.weight, np.asarray([12.5, 12.5, .005, .005], np.float32))
            and np.array_equal(env.task_spec.lower, [-1, -1, -8, -8])
            and np.array_equal(env.task_spec.upper, [1, 1, 8, 8]), "original public task mapping")
    states = env.runtime.snapshot()
    require(set(states) == set(WORLD_KEYS), "complete initial worlds")
    q, v = [], []
    for key, template in zip(WORLD_KEYS, protocol["held_out_templates"], strict=True):
        state = states[key]
        require(type(state) is MuJoCoState and state.world == key and state.tick == 0
                and state.time == 0 and state.last_step is None, "fresh public entry state")
        require(np.array_equal(state.q, np.asarray(template[0], np.float32).astype(np.float64))
                and np.array_equal(state.v, np.asarray(template[1], np.float32).astype(np.float64)),
                "original FP32 held-out template bytes")
        q.append(state.q)
        v.append(state.v)
    require(env.current_frame.ticks == (0,)*8 and env.current_frame.times == (0.,)*8
            and not env.episode_length.any() and not env.episode_id.any(), "fresh task/episode clocks")
    return np.asarray(q), np.asarray(v), states


def source_files():
    values = learning_api()._sources()
    for filename in ("diagnose_mujoco_tyx_reachability.py", "evaluate_mujoco_tyx_initialization_ablation.py"):
        path = Path(__file__).with_name(filename).resolve()
        values[str(path)] = sha256(path.read_bytes()).hexdigest()
    return values


def run_diagnostic(evidence_root, ablation_root, output):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    writer = evidence.EvidenceWriter(destination / "trace.jsonl.gz")
    phase, owner_exit, result = "metadata", None, None
    try:
        protocol = protocol_binding()
        history = load_history(evidence_root, ablation_root, protocol)
        sources = source_files()
        writer.write("protocol", {"protocol_binding": protocol, "history": history,
                                   "steps": STEPS, "switch": SWITCH, "source_files": sources,
                                   "Q_ATOL": Q_ATOL, "V_ATOL": V_ATOL,
                                   "scope": "one reference window; no training, seed loop, or Newton performance claim"})
        phase = "construct"
        with learning_api().tasks().make_environment("J", count=8, held_out=True) as env:
            try:
                phase = "entry-binding"
                native = native_model_metadata(env, protocol)
                q, v, initial = validate_entry(env, protocol)
                writer.write("entry", {"states": initial, "native_model": native,
                                       "env_binding": env.binding, "capabilities": env.runtime.capabilities,
                                       "counter_start": env.counters})
                manifest = learning_api()._manifest(env, "J", None, "N4e-reference-no-learner")
                manifest.update(schema="n4e-j-reachability-run-v1", source_files=sources,
                                historical_binding=history, reference_steps=STEPS, switch=SWITCH,
                                quality_scope="not an N4a learning-gate evaluation")
                evidence.json_write(destination / "manifest.json", manifest)
                phase = "schedule-preflight"
                schedule = make_schedule(q, v)
                writer.write("schedule", schedule)
                phase = "window"
                result = run_window(env, schedule, writer)
                writer.write("window_complete", result)
            finally:
                # Capture only public detached data before owners close, including failures.
                try:
                    owner_exit = {"counters": evidence.plain_counters(env.counters),
                                  "episode_length": env.episode_length, "episode_id": env.episode_id,
                                  "needs_recovery": env.needs_recovery}
                except Exception as error:
                    owner_exit = {"capture_error_type": type(error).__name__, "capture_error": str(error)}
            phase = "owner-close"
        phase = "final-source-binding"
        final_sources = source_files()
        writer.write("owner_exit", owner_exit)
        writer.write("final_source_files", final_sources)
        require(sources == final_sources, "source changed during the single diagnostic")
        writer.close()
        result.update(
            schema="n4e-j-reachability-summary-v1", protocol_binding=protocol, history=history,
            source_files=sources, evidence=writer.receipt(),
            comparison_scope="historical costs and source hashes; no simultaneous unmodified control",
            tolerances={"q_absolute": Q_ATOL, "v_absolute": V_ATOL},
        )
        evidence.json_write(destination / "summary.json", result)
        evidence.json_write(destination / "publication.json", {
            "summary_sha256": sha256((destination / "summary.json").read_bytes()).hexdigest(),
            "classification": result["classification"],
        })
        return result
    except BaseException as error:
        detail = {
            "classification": "REFERENCE_INFEASIBLE" if isinstance(error, ReferenceInfeasible) else "FAILED_REFERENCE_DIAGNOSTIC",
            "phase": phase, "error_type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "cost": None,
            "owner_exit": owner_exit, "window": getattr(error, "detail", None),
            "newton_performance_claim": False, "learning_classification": "UNCHANGED_FIXED_BUDGET_NOT_LEARNED",
        }
        if not writer.closed:
            try:
                writer.write("failure", detail)
            finally:
                writer.close()
        detail["evidence"] = writer.receipt()
        evidence.json_write(destination / "failure.json", evidence.encode(detail))
        raise
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, help="original N4a evidence root")
    parser.add_argument("--ablation-root", required=True, help="closed run97/ablation directory")
    parser.add_argument("--output", required=True, help="fresh output directory")
    args = parser.parse_args()
    result = run_diagnostic(args.evidence_root, args.ablation_root, args.output)
    print(json.dumps({"classification": result["classification"], "world_steps": result["world_steps"],
                      "cost": result["cost"], "learning_classification": result["learning_classification"]}))


if __name__ == "__main__":
    main()

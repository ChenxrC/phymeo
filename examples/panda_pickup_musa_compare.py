"""Run a frozen pickup student and matched jaw interventions on the MUSA integration runtime."""

import argparse
from collections import defaultdict
from pathlib import Path

import mujoco as mj
import numpy as np

from _panda_grasp_insert_scene import build_scene
from _panda_pickup_loads import PickupLoads
from _panda_pickup_student import PickupStudent
from panda_pickup_from_reach import (
    DECISIONS, DT, OBSERVATION_CONTRACT, REPEAT, PickupFeatures, apply_action, effort, read_handoffs,
)
from panda_state_ppo import _sha, _write
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime


BRANCHES = ("student", "open", "one_sided")
CONTACT_SOURCE = "musa_device_same_solve"


def handoff_reset(handoff):
    q = np.asarray(handoff.qpos, dtype=float).copy()
    quat = q[19:23]
    length = np.linalg.norm(quat)
    if not np.isfinite(length) or abs(length - 1.) > 1e-5:
        raise ValueError("invalid FP32 free-pin quaternion at reach handoff")
    # The native reset declaration requires a double-precision unit quaternion.
    # This rescaling preserves its rotation; the device then quantizes once to FP32.
    quat /= length
    return MuJoCoInitialState(q, handoff.qvel)


def set_observed(model, data, q, v, t):
    data.qpos[:], data.qvel[:], data.time = q, v, t
    mj.mj_kinematics(model, data)
    mj.mj_comPos(model, data)


def solved_row(loads, data, entry, state, target, action, raw):
    r = state.last_step
    if (r is None or r.backend != "mujoco-tyx-musa" or r.world != state.world or r.tick != state.tick
            or entry.tick + 1 != state.tick or r.input_time != entry.time or r.contact_time != r.input_time
            or r.output_time != state.time or not np.array_equal(entry.q, r.input_q)
            or not np.array_equal(entry.v, r.input_v)):
        raise RuntimeError("MUSA contact receipt differs from the accepted input state or clock")
    set_observed(loads.model, data, r.input_q, r.input_v, r.input_time)
    force, normal_world = np.zeros((5, 3, 3)), np.zeros((5, 3, 3))
    normal, count = np.zeros((5, 3)), np.zeros((5, 3), dtype=np.int32)
    for c in r.contacts:
        a, b = map(int, c.geometry_ids)
        if not c.active or (a in loads.part_ids) == (b in loads.part_ids):
            continue
        part, other, sign = (a, b, -1.) if a in loads.part_ids else (b, a, 1.)
        i, j = loads.source_ids.get(other, 4), loads.part_ids[part]
        basis = np.asarray(c.frame, dtype=float).reshape(3, 3)
        wrench = np.asarray(c.wrench_contact_frame, dtype=float)
        on_part = sign * (basis.T @ wrench[:3])
        published = c.force_on_geometry1_world if a == part else c.force_on_geometry2_world
        if not np.allclose(on_part, published, rtol=2e-6, atol=1e-7):
            raise RuntimeError("signed world force disagrees with the device contact publication")
        force[i, j] += on_part
        normal_world[i, j] += sign * wrench[0] * basis[0]
        normal[i, j] += wrench[0]
        count[i, j] += 1
        raw.append(np.r_[entry.tick, a, b, c.distance, c.position, basis.ravel(), wrench])
    wrist, part = data.site_xpos[[loads.wrist_site, loads.part_site]]
    rot = data.site_xmat[loads.wrist_site].reshape(3, 3)
    packet = r.stored_control
    dofs = slice(loads.part_dof, loads.part_dof + 6)
    return dict(time_s=r.contact_time, output_time_s=r.output_time, input_tick=entry.tick, contact_solver_code=2,
                qpos=r.input_q, qvel=r.input_v, ctrl=packet.ctrl, end_qpos=state.q, end_qvel=state.v,
                motor_target=target.copy(), action=action.copy(), finger_opening_m=r.input_q[loads.jaws],
                part_xyz=part, wrist_xyz=wrist, part_in_hand_m=rot.T @ (part - wrist),
                force_world_n=force, normal_world_n=normal_world, normal_n=normal, up_n=force @ loads.up,
                contact_count=count, part_noncontact_applied=np.r_[packet.xfrc_applied[loads.part_body],
                                                                  packet.qfrc_applied[dofs], r.qfrc_actuator[dofs]])


def run(args):
    scene = build_scene(urdf_path=args.urdf)
    if _sha(args.scene) != scene.xml_sha256 or DT != scene.timestep:
        raise ValueError("handoff scene XML or timestep differs from the original motor model")
    comp = scene.compile()
    model = comp.mj_model
    handoffs = read_handoffs(args.handoffs, model, weights_path=args.reach_weights)
    student = PickupStudent.load(args.student)
    if student.observation_contract != OBSERVATION_CONTRACT:
        raise ValueError("student observation contract differs")
    if (not args.world or len(set(args.world)) != len(args.world)
            or any(w < 0 or w >= len(handoffs) for w in args.world)):
        raise ValueError("select distinct world row indices present in the handoff file")
    args.out.mkdir(parents=True, exist_ok=False)
    loads = PickupLoads(model)
    worlds = [(w, branch) for w in args.world for branch in BRANCHES]
    keys = tuple(range(len(worlds)))
    features = {w: PickupFeatures(model, handoffs[w].qpos) for w in args.world}
    data = mj.MjData(model)
    initial = {k: handoff_reset(handoffs[w]) for k, (w, _) in enumerate(worlds)}
    reset_binding = {}
    targets = [handoffs[w].target.copy() for w, _ in worlds]
    biases = [handoffs[w].bias.copy() for w, _ in worlds]
    history, witnesses = [defaultdict(list) for _ in keys], [[] for _ in keys]
    actions = np.zeros((len(keys), 16), np.float32)
    observations = {w: [] for w in args.world}
    with MuJoCoRuntime(comp, mode="musa", initial_states=initial, world_keys=keys,
                       cache_root=args.cache, contact_capacity=64, constraint_capacity=256) as rt:
        before = rt.musa_device_report()
        if "S4000" not in before["device_name"]:
            raise RuntimeError(f"expected S4000 device; received {before['device_name']}")
        state = rt.integration_snapshot()
        for k, (w, _) in enumerate(worlds):
            if (not np.array_equal(state[k].q, np.asarray(initial[k].q, dtype=np.float32))
                    or not np.array_equal(state[k].v, np.asarray(handoffs[w].qvel, dtype=np.float32)) or state[k].time != 0.):
                raise ValueError("device reset differs from its declared FP32 state")
            raw = np.asarray(handoffs[w].qpos)
            if not np.array_equal(state[k].q[:19], raw[:19]):
                raise ValueError("device changed handoff joint or workpiece position")
            reset_binding[w] = dict(raw_quat_wxyz=raw[19:].tolist(), device_quat_wxyz=state[k].q[19:].tolist(),
                                    raw_fp32_qpos_bit_equal=bool(np.array_equal(raw, state[k].q)),
                                    max_abs_quaternion_roundoff=float(np.max(np.abs(raw[19:] - state[k].q[19:]))))
        for decision in range(DECISIONS):
            for group, w in enumerate(args.world):
                k = group * len(BRANCHES)
                current = state[k]
                set_observed(model, data, current.q, current.v, current.time)
                obs = features[w].observe(data, targets[k])
                observations[w].append(np.array(obs, copy=True))
                actions[k] = student.act(obs)
                targets[k] = apply_action(model, targets[k], actions[k])
                for offset, branch in enumerate(BRANCHES[1:], 1):
                    n = k + offset
                    actions[n] = actions[k]
                    if branch == "open":
                        actions[n, 14:] = 1.
                    else:
                        actions[n, 15] = 1.
                    targets[n] = apply_action(model, targets[n], actions[n])
                    targets[n][:14] = targets[k][:14]
            for _ in range(REPEAT):
                controls = {k: MuJoCoControl(effort(model, state[k].q, state[k].v, targets[k], biases[k]))
                            for k in keys}
                rt.set_controls(controls)
                rt.step(DT)
                post = rt.integration_snapshot()
                for k in keys:
                    receipt = post[k].last_step
                    row = solved_row(loads, data, state[k], post[k], targets[k], actions[k], witnesses[k])
                    for field, value in row.items():
                        history[k][field].append(value)
                    if receipt.dynamics is None or receipt.dynamics.time != receipt.input_time:
                        raise RuntimeError("missing previous input's device dynamics bias")
                    biases[k] = np.asarray(receipt.dynamics.bias)[:len(biases[k])].copy()
                state = post
            if (decision + 1) % 32 == 0:
                print(f"S4000 completed {decision+1}/{DECISIONS} decisions across {len(keys)} worlds", flush=True)
        after = rt.musa_device_report()
    done = after["completed_kernels"] - before["completed_kernels"]
    if done <= 0 or done != after["launched_kernels"] - before["launched_kernels"]:
        raise RuntimeError("S4000 completed/issued kernel counts disagree")

    by_world = {}
    for group, w in enumerate(args.world):
        traces, info = {}, {}
        for offset, branch in enumerate(BRANCHES):
            k = group * len(BRANCHES) + offset
            trace = {field: np.asarray(values) for field, values in history[k].items()}
            trace["active_part_contact_records"] = np.asarray(witnesses[k], dtype=float).reshape(-1, 22)
            if branch == "student":
                trace["decision_observation"] = np.stack(observations[w])
            path = args.out / f"world-{w:03d}-{branch}.npz"
            np.savez_compressed(path, **trace)
            traces[branch] = trace
            info[branch] = dict(trace=path.name, sha256=_sha(path), score=loads.score(trace, contact_source=CONTACT_SOURCE))
        comparisons = {}
        for branch, mode in (("open", "open_jaw"), ("one_sided", "one_sided")):
            paired = loads.compare(traces["student"], traces[branch], control_mode=mode, contact_source=CONTACT_SOURCE)
            paired["bit_equal_commanded_arm_targets"] = np.array_equal(traces["student"]["motor_target"][:, :14],
                                                                       traces[branch]["motor_target"][:, :14])
            comparisons[branch] = {key: value for key, value in paired.items() if key not in ("closed", "negative")}
        by_world[str(w)] = dict(branches=info, comparisons=comparisons, handoff_reset=reset_binding[w])
    _write(args.out / "summary.json", dict(
        schema="panda-pickup-musa-paired-v1", worlds=by_world, handoffs_sha256=_sha(args.handoffs),
        student_sha256=_sha(args.student), model_sha256=scene.xml_sha256, device=after["device_name"],
        completed_kernels=done, decisions=DECISIONS, action_repeat=REPEAT, physics_timestep_s=DT,
        numerical_solver=receipt.numerical_solver, contact_law=receipt.contact_law, contact_phase=receipt.contact_phase,
        contact_source=CONTACT_SOURCE, policy_evaluation="Host frozen student with feedback from each accepted MUSA decision state.",
        motion="MUSA FP32 integration, actuator and contact solve. Host native FK provides features and site coordinates.",
        reset="Preserve handoff positions and velocities; express the FP32 free-pin quaternion as a unit "
              "FP64 orientation for native initialization before the device casts it back to FP32.",
        control="Host PD with handoff bias for first interval, then preceding device input bias for each physical step. "
                "Open and joint2-open branches copy the student's arm targets and use independent joint feedback.",
        noncontact_channels="Device receipt stored xfrc_applied, qfrc_applied, and qfrc_actuator at free part.",
        active_part_contact_record_fields=["input_tick", "geom0", "geom1", "distance_m", "position_world_xyz[3]",
                                           "frame_normal_tangent_rows[9]", "wrench_on_geom1_in_contact_frame[6]"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("handoffs", "student", "scene", "out"):
        parser.add_argument("--" + field, required=True, type=Path)
    parser.add_argument("--world", type=int, nargs="+", default=[0])
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--reach-weights", type=Path, help="reach checkpoint bound to the handoff file")
    run(parser.parse_args())

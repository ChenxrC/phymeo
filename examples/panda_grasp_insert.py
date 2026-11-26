"""Observe two Panda insertion trials, driven exclusively through joint motors."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from _panda_grasp_insert_control import ARM_KD, ARM_KP, DURATION_S, GraspInsertController
from _panda_grasp_insert_scene import PAD_NAMES, RAIL_NAMES, build_scene
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime


class Observer:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.names = [model.geom(i).name for i in range(model.ngeom)]
        self.part = {"part_handle", "part_shoulder", "assembly_pin"}

    def measure(self, state, last_ctrl, phase):
        q, v = np.asarray(state.q), np.asarray(state.v)
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        site = self.data.site
        entry = site("assembly_entry")
        tip = site("assembly_pin_tip")
        basis = entry.xmat.reshape(3, 3)
        tip_local = basis.T @ (tip.xpos - entry.xpos)
        cos_tilt = basis[:, 2] @ tip.xmat.reshape(3, 3)[:, 2]
        pad = np.zeros(2)
        pad_up = support = seat = other_up = gap = 0.
        contacts = () if state.last_step is None else state.last_step.contacts
        for contact in contacts:
            a, b = (self.names[int(i)] for i in contact.geometry_ids)
            if (a in self.part) == (b in self.part):
                continue
            part, other, sign = (a, b, -1.) if a in self.part else (b, a, 1.)
            wall = other.startswith("board_wall_")
            if wall:
                gap = min(gap, float(contact.distance))
            wrench = np.asarray(contact.wrench_contact_frame)
            world_force = sign * (np.asarray(contact.frame).reshape(3, 3).T @ wrench[:3])
            if other in PAD_NAMES and part == "part_handle":
                pad[PAD_NAMES.index(other)] += max(0., wrench[0])
                pad_up += world_force[2]
            elif other in RAIL_NAMES and part == "part_shoulder":
                support += world_force[2]
            elif wall and part == "part_shoulder":
                seat += world_force[2]
            else:
                other_up += world_force[2]
        return dict(time_s=float(state.time), qpos=q.copy(), qvel=v.copy(), phase=phase,
                    ctrl=np.asarray(last_ctrl).copy(), wrist_xyz=site("right_grasp_center").xpos.copy(),
                    part_xyz=site("part_grasp_center").xpos.copy(), pad_forces=pad, pad_up_n=pad_up,
                    support_n=support, seat_n=seat, other_up_n=other_up, hole_gap_m=gap,
                    depth_m=float(tip_local[2]), lateral_m=float(np.max(np.abs(tip_local[:2]))),
                    tilt_rad=float(np.arccos(np.clip(cos_tilt, -1., 1.))))


def experiment(mode, urdf, cache, out):
    out.mkdir(parents=True, exist_ok=False)
    scene = build_scene(urdf_path=urdf)
    (out / "scene.xml").write_bytes(scene.xml.encode())
    comp = scene.compile()
    model = comp.mj_model
    dt = scene.timestep
    ticks = round(DURATION_S / dt)
    keys = (0, 1, 2)
    controllers = [GraspInsertController(scene, model, open_jaw=bool(k)) for k in keys]
    history = [defaultdict(list) for k in keys]
    last = [np.zeros(16) for k in keys]
    arm_target = [np.asarray(scene.initial_qpos[:14]) for k in keys]
    bias = [np.zeros(14) for k in keys]
    step_input = [np.asarray(scene.initial_qpos) for k in keys]
    phase = "initial"
    observer = Observer(model)
    opts = dict(cache_root=cache, contact_capacity=64, constraint_capacity=256) if mode == "musa" else {}
    init = {k: scene.initial_state() for k in keys}
    with MuJoCoRuntime(comp, mode=mode, initial_states=init, world_keys=keys, **opts) as rt:
        snapshot = rt.integration_snapshot if mode == "musa" else rt.snapshot
        before = rt.musa_device_report() if mode == "musa" else {}
        for tick in range(ticks + 1):
            states = snapshot()
            for k in keys:
                row = observer.measure(states[k], last[k], phase)
                report = states[k].last_step
                force_time = 0. if report is None else report.input_time
                if report is not None and report.contact_time is not None:
                    force_time = report.contact_time
                row.update(arm_target_q=arm_target[k], arm_bias_nms=bias[k],
                           contact_qpos=step_input[k], contact_phase=phase,
                           contact_time_s=float(force_time))
                for name, value in row.items():
                    history[k][name].append(value)
            if tick == ticks:
                break
            chosen = controllers[0].advance(states[0].q, states[0].v, tick * dt)
            target = np.asarray(chosen.diagnostics["arm_target_q"])
            opened = controllers[1].advance(states[1].q, states[1].v, tick * dt, reference_arm_target=target)
            torque_replay = controllers[2].advance(states[2].q, states[2].v, tick * dt, reference_ctrl=chosen.ctrl)
            if phase != chosen.phase or tick % 500 == 0:
                print(f"{tick:5d} {chosen.phase:10s} xyz={history[0]['part_xyz'][-1]} "
                      f"pad={history[0]['pad_forces'][-1]} depth={history[0]['depth_m'][-1]:.5f}", flush=True)
            last = [chosen.ctrl, opened.ctrl, torque_replay.ctrl]
            arm_target = [target, np.asarray(opened.diagnostics["arm_target_q"]), np.zeros(14)]
            bias = [controller.data.qfrc_bias[:14].copy() for controller in controllers]
            step_input = [np.array(states[k].q, copy=True) for k in keys]
            phase = chosen.phase
            rt.set_controls({k: MuJoCoControl(last[k]) for k in keys})
            rt.step(dt)
        after = rt.musa_device_report() if mode == "musa" else {}
    if mode == "musa" and ("S4000" not in after["device_name"] or
                           after["completed_kernels"] <= before["completed_kernels"] or
                           after["launched_kernels"] != after["completed_kernels"]):
        raise RuntimeError("S4000 device execution could not be verified")
    traces = [{name: np.asarray(values) for name, values in item.items()} for item in history]
    hashes = {}
    for label, trace in zip(("closed", "open", "open-torque"), traces, strict=True):
        path = out / (label + ".npz")
        trace.update(arm_kp=ARM_KP, arm_kd=ARM_KD, arm_ctrlrange=controllers[0].limits[:14])
        np.savez_compressed(path, **trace, model_sha256=scene.xml_sha256)
        hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    weight = float(model.body("assembly_part").mass[0]) * np.linalg.norm(model.opt.gravity)
    from _panda_grasp_insert_metrics import evaluate, evaluate_position_control

    score = evaluate_position_control(traces[0], traces[1], dt, float(weight))
    torque_score = evaluate(traces[0], traces[2], dt, float(weight))
    result = dict(score=score, torque_replay=torque_score, weight_n=float(weight), steps_per_world=ticks, timestep_s=dt,
                  seconds=scene.timestep*ticks, model_sha256=scene.xml_sha256, trace_sha256=hashes,
                  backend=mode, device=after.get("device_name", "cpu"),
                  completed_kernels=after.get("completed_kernels", 0) - before.get("completed_kernels", 0),
                  control="joint motors only; open target replay with feedback and separate open torque replay")
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print("task success", score["success"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("musa", "cpu"), default="musa")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    experiment(args.mode, args.urdf, args.cache, args.out)

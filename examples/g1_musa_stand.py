"""Short paired G1 standing run on MUSA and native MuJoCo, with a two-step timing probe."""

import argparse
import hashlib
import json
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import g1_mujoco_tyx_policy as g1
import mujoco as mj
import numpy as np
import prepare_g1_stand_assets as assets_spec

from phymeo.backends.mujoco_tyx import _kernels

H = 0.002
MIN_STAND_STEPS = 250
DRIFT_LIMITS = {"root_position_m": 0.02, "root_angle_rad": 0.10, "joint_rms_rad": 0.05}


def pose_difference(protocol, gpu_q, cpu_q):
    # The two signs of a unit quaternion describe the same orientation.
    a, b = gpu_q[3:7], cpu_q[3:7]
    dot = abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "root_position_m": float(np.linalg.norm(gpu_q[:3] - cpu_q[:3])),
        "root_angle_rad": float(2 * np.arccos(np.clip(dot, 0, 1))),
        "joint_rms_rad": float(np.sqrt(np.mean((gpu_q[protocol.qa] - cpu_q[protocol.qa]) ** 2))),
    }


def forbid_host_dynamics(*args, **kwargs):
    raise RuntimeError("MUSA step invoked CPU dynamics, host contact forces or another TYX backend")


def run(args, journal):
    if mj.__version__ != "3.3.7":
        raise ValueError("this comparison requires the original MuJoCo 3.3.7 oracle")
    sources = {name: g1.binding(args.assets / name) for name in (g1.ASSET_NAME, "deploy.yaml")}
    yaml = (args.assets / "deploy.yaml").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(yaml).hexdigest() != assets_spec.DEPLOY_LF_SHA:
        raise ValueError("G1 deployment parameters differ from the pinned Unitree source")
    compilation, model, _ = g1.compile_source(args.assets, H)
    protocol = g1.Protocol(model, args.assets, stand_only=True)
    q, v = protocol.initial()
    gpu_stats, cpu_stats = (g1.Stats(protocol, H, stand=True) for _ in range(2))
    maximum = dict.fromkeys(DRIFT_LIMITS, 0.0)
    journal.write(
        "sources",
        durable=True,
        inputs=sources,
        step_s=H,
        steps=args.steps,
        drift_limits=DRIFT_LIMITS,
        settling_time_s=g1.SETTLING,
    )
    start = time.perf_counter()
    with g1.runtime_for(compilation, q, v, 0, "musa", args.cache_root) as runtime:
        first = runtime.musa_device_report()
        caps = runtime.capabilities
        if not (
            caps["gpu"] and first["backend"] == "musa" and first["device_name"] and first["arch"]
        ):
            raise ValueError("could not identify an actual MUSA device")
        if (caps["possible_contact_pairs"], caps["candidate_slots"]) != (502, 942):
            raise ValueError("unexpected G1 contact candidate layout")
        state = runtime.integration_snapshot()[0]
        cpu = mj.MjData(model)
        cpu.qpos[:], cpu.qvel[:], cpu.time = state.q, state.v, state.time
        gpu_control, cpu_control = (
            g1.Controller(protocol, state.q, H, stand=True) for _ in range(2)
        )
        feet = g1.endpoint_foot_centers(protocol, state.q)
        for stats in (gpu_stats, cpu_stats):
            stats.foot_origin = feet[:, :2].copy()
        previous = first
        journal.write(
            "initial",
            durable=True,
            device=first,
            model_dims=[model.nq, model.nv, model.nu],
            possible_pairs=502,
            slots=942,
            q=state.q,
            v=state.v,
            initialization_wall_s=time.perf_counter() - start,
        )
        for tick in range(args.steps):
            step_start = time.perf_counter()
            proposal = gpu_control.propose(state.q, state.v, state.time, tick)
            requested = g1.packet_for(protocol, proposal, tick * H, 0)
            runtime.set_controls({0: requested})
            product_before = runtime.counters["completed_kernels"]
            with ExitStack() as guard:
                for name in ("mj_step", "mj_forward", "mj_contactForce"):
                    guard.enter_context(patch.object(mj, name, forbid_host_dynamics))
                guard.enter_context(
                    patch.object(type(_kernels.advance), "__call__", forbid_host_dynamics)
                )
                runtime.step(H)
                after = runtime.integration_snapshot()[0]
            gpu_wall = time.perf_counter() - step_start
            report = runtime.musa_device_report()
            row = after.last_step
            actual_launches = report["completed_kernels"] - previous["completed_kernels"]
            if not (
                row.backend == "mujoco-tyx-musa"
                and row.tick == after.tick == tick + 1
                and row.contact_time == state.time
                and report["active_bank"] != previous["active_bank"]
                and actual_launches == runtime.counters["completed_kernels"] - product_before > 0
            ):
                raise RuntimeError(
                    "MUSA completion counters, state bank or input-phase receipt differ"
                )
            if abs(after.time - after.tick * float(np.float32(H))) > 2e-6:
                raise ValueError("accepted device time does not match the step count")
            previous = report
            g1.input_checks(row, state, g1.stored_packet(requested))
            if not g1.actuator_check(protocol, row)["passed"]:
                raise ValueError("published actuator torques differ from the capped PD controls")
            gpu_control.commit(proposal)
            gpu_loads = g1.support(protocol, row.contacts, row.input_v, row.dynamics.body_com)
            _, _, gpu_fail, _ = gpu_stats.update(
                after.q,
                after.v,
                after.time,
                state.time,
                gpu_loads,
                g1.receipt_energy(model, row),
                g1.endpoint_foot_centers(protocol, after.q),
            )
            # This reference feeds back its own state and never receives a subsequent device pose.
            cpu_q, cpu_v, cpu_t = cpu.qpos.copy(), cpu.qvel.copy(), cpu.time
            cpu_proposal = cpu_control.propose(cpu_q, cpu_v, cpu_t, tick)
            cpu_packet = g1.packet_for(protocol, cpu_proposal, tick * H, 0)
            g1.set_native(cpu, cpu_packet)
            mj.mj_step(model, cpu)
            cpu_control.commit(cpu_proposal)
            if abs(cpu.time - (tick + 1) * H) > 2e-12 or any(w.number for w in cpu.warning):
                raise ValueError(
                    "independent native reference produced a warning or unexpected time"
                )
            native = g1.native_fields(model, cpu, cpu_q, cpu_v, cpu_packet, full=False)
            cpu_loads = g1.support(protocol, native["contacts"], cpu_v, native["body_com"])
            _, _, cpu_fail, _ = cpu_stats.update(
                cpu.qpos,
                cpu.qvel,
                cpu.time,
                cpu_t,
                cpu_loads,
                native["energy"],
                g1.endpoint_foot_centers(protocol, cpu.qpos),
            )
            difference = pose_difference(protocol, after.q, cpu.qpos)
            for key, value in difference.items():
                maximum[key] = max(maximum[key], value)
            failures = [
                k for k, v in difference.items() if not np.isfinite(v) or v > DRIFT_LIMITS[k]
            ]
            journal.write(
                "step",
                tick=tick + 1,
                gpu_q=after.q,
                gpu_v=after.v,
                cpu_q=cpu.qpos,
                cpu_v=cpu.qvel,
                gpu_control=requested.ctrl,
                cpu_control=cpu_packet.ctrl,
                gpu_load=gpu_loads,
                cpu_load=cpu_loads,
                difference=difference,
                completed_kernels=report["completed_kernels"],
                gpu_step_wall_s=gpu_wall,
            )
            if tick < 2 or (tick + 1) % 25 == 0 or tick + 1 == args.steps:
                print(
                    f"G1 {tick + 1}/{args.steps} t={after.time:.3f}s "
                    f"MUSA step {gpu_wall:.3f}s, elapsed {time.perf_counter() - start:.2f}s",
                    file=sys.stderr,
                    flush=True,
                )
            if gpu_fail or cpu_fail or failures:
                raise ValueError(
                    f"trajectory check failed: MUSA={gpu_fail}, CPU={cpu_fail}, drift={failures}"
                )
            state = after
        assessed = args.steps >= MIN_STAND_STEPS
        gpu = gpu_stats.summary(assessed=assessed)
        reference = cpu_stats.summary(assessed=assessed)
        failures = gpu["checkpoint_failures"] + reference["checkpoint_failures"]
        return {
            "status": "FAIL" if failures else ("STAND_PASS" if assessed else "PROBE_ONLY"),
            "steps": args.steps,
            "simulated_s": H * args.steps,
            "wall_s": time.perf_counter() - start,
            "failures": failures,
            "gpu": gpu,
            "independent_cpu": reference,
            "max_pose_difference": maximum,
            "musa_start": first,
            "musa_end": previous,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets", type=Path, required=True, help="directory with pinned G1 XML and deploy.yaml"
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new .jsonl or .jsonl.gz path")
    parser.add_argument(
        "--steps", type=int, default=MIN_STAND_STEPS, help="2 for a probe; 250 for standing"
    )
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("--steps must be 1..1000 (at most two simulated seconds)")
    journal = g1.Journal(args.output)
    try:
        try:
            summary = run(args, journal)
        except Exception as error:
            summary = {"status": "FAIL", "error": str(error), "traceback": traceback.format_exc()}
        journal.finish(summary)
        print(json.dumps(g1.json_value(summary), ensure_ascii=False, allow_nan=False), flush=True)
        return int(summary["status"] == "FAIL")
    finally:
        journal.close()


if __name__ == "__main__":
    raise SystemExit(main())

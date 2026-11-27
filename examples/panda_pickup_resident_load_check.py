"""Compare resident contact loads against MUSA integration snapshots on matched supported inputs."""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np

from _panda_grasp_insert_scene import build_scene
from _panda_pickup_loads import BEARING_WEIGHT, PickupLoads
from _panda_pickup_rl_reward import ResidentPickupLoads
from panda_pickup_from_reach import DT, PUBLISHED, effort, read_handoffs
from panda_pickup_musa_compare import handoff_reset, solved_row
from panda_state_ppo import _sha, _write, require_musa
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime


def compare_arrays(reference, resident, weight_n):
    """One simulator, two public receipt paths. Require matched states and witnessed support."""
    for k in ("qpos", "qvel", "ctrl", "end_qpos", "end_qvel"):
        np.testing.assert_array_equal(reference[k], resident[k], err_msg=f"physical calibration differs for {k}")
    np.testing.assert_allclose(reference["time_s"], resident["time_s"], rtol=0., atol=1e-9)
    errors = {}
    for k in ("normal_n", "up_n", "force_world_n"):
        np.testing.assert_allclose(resident[k], reference[k], atol=5e-6, rtol=5e-6,
                                   err_msg=f"resident contact aggregate differs for {k}")
        errors[k] = float(np.max(np.abs(resident[k] - reference[k])))
    rail_load = reference["up_n"][:, 2:4].sum(axis=(1, 2))
    support = int(np.count_nonzero(rail_load >= BEARING_WEIGHT * weight_n))
    if not support:
        raise RuntimeError("Calibration trajectory did not contain a fork-supported physical solve")
    return dict(max_absolute_errors_n=errors, supported_steps=support, peak_fork_up_n=float(rail_load.max()),
                weight_n=weight_n, load_atol_n=5e-6, load_rtol=5e-6, input_and_successor_state_bit_equal=True)


def run(args):
    torch = require_musa()
    torch.musa.set_device(0)
    device = torch.device("musa:0")
    name = torch.musa.get_device_name(0)
    if "S4000" not in name or not 0 <= args.world < 12 or args.steps < 1:
        raise ValueError("Expected the S4000, a published world row and a positive physical step count")
    scene = build_scene(urdf_path=args.urdf)
    if scene.xml_sha256 != _sha(PUBLISHED / "scene.xml") or scene.timestep != DT:
        raise ValueError("Calibration requires the published physical model")
    compiled = scene.compile()
    model, initial = compiled.mj_model, {}
    loads = PickupLoads(model)
    handoff = read_handoffs(args.handoffs, model)[args.world]
    initial[0] = handoff_reset(handoff)
    options = dict(mode="musa", initial_states=initial, world_keys=(0,), cache_root=args.cache,
                   contact_capacity=64, constraint_capacity=256)
    args.out.mkdir(parents=True, exist_ok=False)
    data = mj.MjData(model)
    keys = ("time_s", "qpos", "qvel", "ctrl", "end_qpos", "end_qvel", "normal_n", "up_n", "force_world_n")
    ref = {k: [] for k in keys}
    with MuJoCoRuntime(compiled, **options) as rt:
        current = rt.integration_snapshot()[0]
        bias = handoff.bias
        for _ in range(args.steps):
            ctrl = np.asarray(effort(model, current.q, current.v, handoff.target, bias), np.float32)
            rt.set_controls({0: MuJoCoControl(ctrl)})
            rt.step(DT)
            successor = rt.integration_snapshot()[0]
            row = solved_row(loads, data, current, successor, handoff.target, np.zeros(16), [])
            np.testing.assert_array_equal(row["ctrl"], ctrl)
            if np.any(row["part_noncontact_applied"]):
                raise ValueError("Integration receipt contains direct free-part actuation or external load")
            for k in keys:
                ref[k].append(np.array(row[k], copy=True))
            bias = successor.last_step.dynamics.bias[:16]
            current = successor
    ref = {k: np.stack(v) for k, v in ref.items()}
    resident = {k: [] for k in keys}
    with MuJoCoRuntime(compiled, **options) as rt:
        with rt.musa_resident() as bank:
            bank.reset()
            reader = ResidentPickupLoads(torch, model, bank, device)
            begin = bank.report()
            buffers = bank.buffers()
            for command in ref["ctrl"]:
                q, v, clock = (buffers[k].to_torch() for k in ("qpos", "qvel", "time"))
                command_t = torch.as_tensor(command, dtype=torch.float32, device=device).reshape(1, 16)
                buffers["ctrl"].copy_from_torch(command_t)
                # Retain actual submitted motor buffer, independently of command_t.
                ctrl = buffers["ctrl"].to_torch()
                buffers = bank.step()
                normal, up = reader.read()
                world_force = reader.world_force()
                for k, batch in (("qpos", q), ("qvel", v), ("clock", clock), ("ctrl", ctrl),
                                 ("end_qpos", buffers["qpos"].to_torch()), ("end_qvel", buffers["qvel"].to_torch()),
                                 ("normal_n", normal), ("up_n", up), ("force_world_n", world_force)):
                    if k == "clock":
                        # Keep the native compensated pair for FP64 reconstruction at the boundary.
                        resident["time_s"].append(batch[0].clone())
                    else:
                        resident[k].append(batch[0].clone())
            resident = {k: torch.stack(v).cpu().numpy() for k, v in resident.items()}
            clock = resident["time_s"].astype(float)
            resident["time_s"] = clock[:, 0] - clock[:, 1]
            end = bank.report()
            if (end["accepted_world_steps"]-begin["accepted_world_steps"] != args.steps
                    or end["explicit_readback_bytes"] != begin["explicit_readback_bytes"]
                    or end["downloaded_bytes"]-begin["downloaded_bytes"] != 4*args.steps):
                raise ValueError("Calibration left the expected one-status-word resident physics path")
    arrays = {f"{prefix}_{k}": v for prefix, rows in (("integration", ref), ("resident", resident))
              for k, v in rows.items()}
    trace = args.out / "contact-comparison.npz"
    np.savez_compressed(trace, **arrays)
    result = dict(schema="panda-pickup-resident-load-calibration-v1", trace_sha256=_sha(trace), device=name,
                  handoff_sha256=_sha(args.handoffs), world=args.world, physical_steps=args.steps,
                  model_sha256=scene.xml_sha256, resident_start=begin, resident_end=end)
    try:
        result.update(compare_arrays(ref, resident, loads.weight_n), status="passed")
    except (RuntimeError, AssertionError) as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        _write(args.out / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoffs", type=Path, default=PUBLISHED.parent / "panda-pickup-from-reach-s4000/factor-update.npz")
    parser.add_argument("--world", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())

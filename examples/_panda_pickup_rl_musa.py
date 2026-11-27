"""On-policy pickup rollouts using the S4000's accepted resident physics and contact solves."""

import hashlib
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from _panda_pickup_loads import PickupLoads
from _panda_pickup_rl_actor import pickup_features
from _panda_pickup_rl_evaluate import CONTACT_SOURCE, ResidentPickupTrace, _check_motor_model
from _panda_pickup_rl_reward import PickupRLReward, ResidentPickupLoads
from _panda_ppo_reach import ReachTask, _angle, _mm
from panda_pickup_from_reach import DT, KD, KP, sha
from panda_pickup_musa_compare import handoff_reset
from panda_state_ppo import model_digest, require_musa
from phymeo.backends.mujoco_runtime import MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml


def _transport(begin, end, steps, worlds, state_bytes):
    fields = ("resident_steps", "accepted_world_steps", "resident_resets", "failed_batches",
              "uploaded_bytes", "downloaded_bytes", "status_downloaded_bytes", "explicit_readback_bytes",
              "torch_to_tyx_bytes", "tyx_to_torch_bytes", "tyx_to_tyx_bytes", "device_to_device_bytes",
              "launched_kernels", "completed_kernels")
    delta = {k: int(end[k]-begin[k]) for k in fields}
    expected = dict(resident_steps=steps, accepted_world_steps=steps*worlds, resident_resets=0, failed_batches=0,
                    uploaded_bytes=0, downloaded_bytes=4*steps*worlds, status_downloaded_bytes=4*steps*worlds,
                    explicit_readback_bytes=0, torch_to_tyx_bytes=4*16*steps*worlds,
                    tyx_to_torch_bytes=state_bytes*steps)
    for name, amount in expected.items():
        if delta[name] != amount:
            raise RuntimeError(f"resident transfer/step accounting differs at {name}: {delta[name]} vs {amount}")
    if (delta["completed_kernels"] <= 0 or delta["launched_kernels"] != delta["completed_kernels"]
            or delta["device_to_device_bytes"] != sum(delta[k] for k in
                    ("torch_to_tyx_bytes", "tyx_to_torch_bytes", "tyx_to_tyx_bytes"))):
        raise RuntimeError("inconsistent resident kernel completion or device-copy counters")
    return delta


@torch.no_grad()
def run_musa(domain, policy, *, decisions=160, repeat=8, mode="train", trace=False, cache=None):
    """Return device PPO tensors, independent per-world evaluation scores, and a JSON-ready summary."""
    if mode not in ("train", "prior", "candidate"):
        raise ValueError("mode must be train, prior or candidate")
    if type(decisions) is not int or type(repeat) is not int or min(repeat, decisions) < 1:
        raise ValueError("expected positive integer decision and repeat counts")
    require_musa()
    device = next(policy.parameters()).device
    if device.type != "musa":
        raise ValueError("the resident learner and frozen policies must be on MUSA")
    steps, starts = decisions*repeat, domain.handoffs
    worlds = len(starts)
    capture = mode != "train" or bool(trace)
    if not worlds or (capture and steps < 2):
        raise ValueError("a nonempty world batch and at least two solved inputs for traces are required")
    compiled = compile_mujoco_xml(domain.xml, require_asset=False)
    model = compiled.mj_model
    digest = model_digest(model)
    if (digest != model_digest(domain.model) or model.opt.timestep != DT
            or model.opt.integrator != mj.mjtIntegrator.mjINT_EULER):
        raise ValueError("compiled pickup model differs from the declared 2 ms Euler domain")
    loads = PickupLoads(model)
    _check_motor_model(model, loads)
    joints = model.actuator_trnid[:, 0]
    if (not np.array_equal(model.jnt_qposadr[joints], np.arange(16))
            or not np.array_equal(model.actuator_gear, np.tile([1., 0., 0., 0., 0., 0.], (16, 1)))):
        raise ValueError("pickup requires the ordered sixteen unit-gear joint motors")
    fk = ReachTask(torch, model, device)
    reward_fn = PickupRLReward(torch, loads.weight_n, DT, up=loads.up, device=device)
    keys = tuple(range(worlds))
    initial = {w: handoff_reset(starts[w]) for w in keys}

    def tensor(array):
        return torch.as_tensor(np.ascontiguousarray(array), device=device, dtype=torch.float32)

    target = tensor(np.stack([s.target for s in starts]))
    first_bias = tensor(np.stack([s.bias for s in starts]))
    limits = np.array(model.jnt_range[joints], copy=True)
    limits[:14] += [.001, -.001]
    low, high = (tensor(col) for col in limits.T)
    effort_limit = np.array(model.actuator_ctrlrange, copy=True)
    effort_limit[14:, 0] = np.maximum(effort_limit[14:, 0], -20.)
    effort_limit[14:, 1] = np.minimum(effort_limit[14:, 1], 20.)
    effort_low, effort_high = (tensor(col) for col in effort_limit.T)
    kp, kd = tensor(KP), tensor(KD)
    trace_dir = Path(trace) if isinstance(trace, (str, Path)) else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=False)
    records = {k: [] for k in ("obs", "raw", "logp", "value", "reward")} if mode == "train" else {}
    returns = torch.zeros(worlds, dtype=torch.float32, device=device)
    with MuJoCoRuntime(compiled, mode="musa", initial_states=initial, world_keys=keys, cache_root=cache,
                       contact_capacity=64, constraint_capacity=256) as runtime, runtime.musa_resident() as bank:
        bank.reset()
        meta = bank.report()
        name = str(meta["device_name"])
        if (meta["device_index"] != device.index or "S4000" not in name or "S4000" not in torch.musa.get_device_name(device.index)
                or bank.world_keys != keys or bank.timestep != DT):
            raise RuntimeError("actual S4000 or ordered resident pickup world batch differs")
        buffers = bank.buffers()
        q, v, ctrl = (torch.empty(buffers[k].shape, dtype=torch.float32, device=device)
                      for k in ("qpos", "qvel", "ctrl"))
        bias = torch.zeros_like(v)
        bias[:, :16].copy_(first_bias)
        clock = torch.empty(buffers["time"].shape, dtype=torch.float32, device=device) if capture else None
        buffers["qpos"].copy_to_torch(q)
        buffers["qvel"].copy_to_torch(v)
        if capture:
            buffers["time"].copy_to_torch(clock)
        if (not torch.equal(q[:, :23], tensor(np.stack([initial[w].q for w in keys])))
                or not torch.equal(v[:, :22], tensor(np.stack([initial[w].v for w in keys])))):
            raise RuntimeError("resident reset differs from declared physical handoff")
        poses = fk.poses(q[:, :23])
        origin = poses[2][0].clone()
        reward_fn.reset(origin)
        reader = ResidentPickupLoads(torch, model, bank, device)
        recorder = ResidentPickupTrace(model, bank, steps) if capture else None
        begin = bank.report()
        for decision in range(decisions):
            features = pickup_features(torch, poses, q[:, :23], target, origin[:, 2])
            obs = policy.observation(features, v[:, :22])
            if mode == "prior":
                action = policy.prior_action(obs[:, :21])
            else:
                action, raw, logp, value = policy.sample(obs, deterministic=mode == "candidate")
                if mode == "train":
                    for k, x in (("obs", obs), ("raw", raw), ("logp", logp), ("value", value)):
                        records[k].append(x)
            target[:, :14] += .02 * action[:, :14]
            target[:, 14:] = .0175 * (action[:, 14:] + 1.)
            target = torch.maximum(low, torch.minimum(high, target))
            decision_reward = torch.zeros_like(returns)
            for substep in range(repeat):
                # q, v, poses and clock remain the physical input until the
                # accepted contact forces have been measured and recorded.
                _, (wrist, hand_r), (part, pin_r) = poses
                local = (hand_r * (part-wrist).unsqueeze(-1)).sum(-2)
                angle = _angle(torch, _mm(hand_r.transpose(-1, -2), pin_r))
                effort = kp*(target-q[:, :16]) - kd*v[:, :16] + bias[:, :16]
                ctrl.copy_(torch.maximum(effort_low, torch.minimum(effort_high, effort)))
                buffers["ctrl"].copy_from_torch(ctrl)
                buffers = bank.step()
                normal, vertical = reader.read()
                step_reward, _ = reward_fn.step(normal, vertical, part, wrist, local, angle)
                decision_reward += step_reward
                if recorder is not None:
                    recorder.append(decision*repeat + substep, clock, q[:, :23], v[:, :22], ctrl, target, action,
                                    normal, vertical, reader.world_force(), part, wrist, local)
                buffers["qpos"].copy_to_torch(q)
                buffers["qvel"].copy_to_torch(v)
                buffers["entry_bias"].copy_to_torch(bias)
                if capture:
                    buffers["time"].copy_to_torch(clock)
                if decision != decisions-1 or substep != repeat-1:
                    poses = fk.poses(q[:, :23])
            returns += decision_reward
            if mode == "train":
                records["reward"].append(decision_reward)
        end = bank.report()
        reads = [q, v, bias, *reader.raw.values()]
        if capture:
            reads.append(clock)
        state_bytes = sum(x.numel()*x.element_size() for x in reads)
        counters = _transport(begin, end, steps, worlds, state_bytes)
        if not bool((torch.isfinite(returns).all() & torch.isfinite(q).all() & torch.isfinite(v).all()).cpu()):
            raise RuntimeError("nonfinite resident pickup trajectory")
        traces, scores = recorder.finish() if capture else ([], [])
    if mode == "train":
        scores = []
    summary = dict(backend="mujoco-tyx-musa-resident", mode=mode, worlds=worlds, decisions=decisions, repeat=repeat,
                   timestep_s=DT, physical_steps_per_world=steps, physics_world_steps=steps*worlds,
                   policy_transitions=decisions*worlds, mean_return=float(returns.mean().cpu()),
                   returns=returns.cpu().tolist(), reward_success_worlds=int(reward_fn.succeeded.sum().cpu()),
                   scored_success_worlds=sum(s["success"] for s in scores) if scores else None,
                   contact_source=CONTACT_SOURCE, policy_device=str(device), device_name=name,
                   model_sha256=hashlib.sha256(domain.xml.encode()).hexdigest(),
                   frozen_model_sha256=digest, transport=counters)
    if trace_dir is not None:
        files = []
        for w, trace_arrays in enumerate(traces):
            path = trace_dir / f"world-{w:03d}.npz"
            np.savez_compressed(path, **trace_arrays)
            files.append(dict(file=path.name, sha256=sha(path)))
        summary["trace_files"] = files
    elif trace:
        summary["traces"] = [{k: a.tolist() for k, a in row.items()} for row in traces]
    return {k: torch.stack(values) for k, values in records.items()}, scores, summary

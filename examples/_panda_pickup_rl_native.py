"""Native CPU pickup rollout with input-state MuJoCo contact forces and batched Torch decisions."""

from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from _panda_pickup_loads import PickupLoads
from _panda_pickup_rl_reward import PickupRLReward
from panda_pickup_from_reach import DT, PickupFeatures, apply_action, effort, sha


def _step_input(model, data, loads, target, bias):
    """One Euler step. The cached sites, bias and constraint wrenches belong to the input state."""
    t, q, v = data.time, data.qpos.copy(), data.qvel.copy()
    data.ctrl[:] = effort(model, q, v, target, bias)
    mj.mj_step(model, data)
    if abs(data.time - (t + DT)) > 1e-10:
        raise RuntimeError("MuJoCo failed to advance a full pickup timestep")
    # mj_step populated the single input solve, then advanced q/v/time. Preserve
    # its cached FK and forces; substitute the co-timed input integration fields.
    row = loads.sample(data)
    row.update(time_s=t, qpos=q, qvel=v, finger_opening_m=q[loads.jaws])
    hand_r, pin_r = data.site_xmat[[loads.wrist_site, loads.part_site]].reshape(2, 3, 3)
    rel = hand_r.T @ pin_r
    sine = np.linalg.norm(rel - rel.T) / np.sqrt(8.)
    orientation = np.arctan2(sine, .5 * (np.trace(rel) - 1.))
    return row, orientation, data.qfrc_bias[:16].copy()


@torch.no_grad()
def run_native(domain, policy, *, decisions=160, repeat=8, mode="train", trace=False):
    """Return PPO tensors for training, independent scores for evaluation, and a JSON-ready summary.

    trace=False retains physical history only as needed for evaluation. A path writes full traces
    to a new directory. True embeds those traces as plain values in the summary.
    """
    if mode not in ("train", "prior", "candidate"):
        raise ValueError("mode must be train, prior or candidate")
    if type(decisions) is not int or type(repeat) is not int or min(decisions, repeat) < 1:
        raise ValueError("positive integer decision and repeat counts required")
    model, starts = domain.model, domain.handoffs
    worlds, steps = len(starts), decisions * repeat
    if not worlds or model.opt.timestep != DT or model.opt.integrator != mj.mjtIntegrator.mjINT_EULER:
        raise ValueError("native pickup needs nonempty handoffs and the original 2 ms Euler integration")
    if mode != "train" and steps < 2:
        raise ValueError("the independent evaluator requires at least two solved inputs")
    if next(policy.parameters()).device.type != "cpu":
        raise ValueError("native reference requires the Torch CPU pickup policy")
    trace_dir = Path(trace) if isinstance(trace, (str, Path)) else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=False)
    loads = PickupLoads(model)
    data = [mj.MjData(model) for _ in starts]
    observers, origin = [], []
    for d, start in zip(data, starts, strict=True):
        d.qpos[:], d.qvel[:] = start.qpos, start.qvel
        mj.mj_kinematics(model, d)
        origin.append(d.site_xpos[loads.part_site].copy())
        observers.append(PickupFeatures(model, start.qpos))
    target = [s.target.copy() for s in starts]
    bias = [s.bias.copy() for s in starts]
    reward_fn = PickupRLReward(torch, loads.weight_n, DT, up=loads.up, device="cpu")
    reward_fn.reset(np.stack(origin))
    records = {key: [] for key in ("obs", "raw", "logp", "value", "reward")} if mode == "train" else {}
    kept = {}  # Allocate only the evaluator's channels unless full traces were requested.
    required = {"time_s", "contact_solver_code", "qpos", "qvel", "ctrl", "finger_opening_m", "part_xyz",
                "wrist_xyz", "part_in_hand_m", "force_world_n", "normal_n", "up_n", "part_noncontact_applied"}
    episode_return = torch.zeros(worlds, dtype=torch.float32)
    for decision in range(decisions):
        for d in data:
            mj.mj_kinematics(model, d)
        features = np.stack([f.observe(d, t) for f, d, t in zip(observers, data, target, strict=True)])
        obs = policy.observation(features, np.stack([d.qvel for d in data]))
        if mode == "prior":
            action = policy.prior_action(obs[:, :21])
        else:
            action, raw, logp, value = policy.sample(obs, deterministic=(mode == "candidate"))
            if mode == "train":
                for k, x in (("obs", obs), ("raw", raw), ("logp", logp), ("value", value)):
                    records[k].append(x)
        action = action.numpy()
        target = [apply_action(model, target[w], action[w]) for w in range(worlds)]
        decision_reward = torch.zeros_like(episode_return)
        for substep in range(repeat):
            tick = decision * repeat + substep
            rows, orientation = [], []
            for w, d in enumerate(data):
                row, angle, bias[w] = _step_input(model, d, loads, target[w], bias[w])
                if np.any(row["part_noncontact_applied"]):
                    raise RuntimeError("unexpected direct load on the free pin")
                rows.append(row)
                orientation.append(angle)
                if mode != "train" or trace:
                    if trace:
                        row.update(motor_target=target[w], action=action[w], input_tick=tick)
                    for key, value in row.items():
                        if not trace and key not in required:
                            continue
                        if key not in kept:
                            a = np.asarray(value)
                            kept[key] = np.empty((worlds, steps, *a.shape), dtype=a.dtype)
                        kept[key][w, tick] = value
            channels = [np.stack([row[key] for row in rows]) for key in
                        ("normal_n", "up_n", "part_xyz", "wrist_xyz", "part_in_hand_m")]
            step_reward, _ = reward_fn.step(*channels, np.asarray(orientation))
            decision_reward += step_reward
        episode_return += decision_reward
        if mode == "train":
            records["reward"].append(decision_reward)
    if not torch.isfinite(episode_return).all() or any(
            not np.isfinite(np.r_[d.qpos, d.qvel]).all() or any(x.number for x in d.warning) for d in data):
        raise RuntimeError("nonfinite pickup episode or native MuJoCo warning")
    scores, files, embedded = [], [], []
    for w in range(worlds):
        phys = {key: value[w] for key, value in kept.items()}
        if mode != "train":
            scores.append(loads.score(phys, contact_source="native_mujoco_cpu"))
        if trace_dir is not None:
            path = trace_dir / f"world-{w:03d}.npz"
            np.savez_compressed(path, **phys)
            files.append(dict(file=path.name, sha256=sha(path)))
        elif trace:
            embedded.append({key: a.tolist() for key, a in phys.items()})
    summary = dict(backend="native_mujoco_cpu", mode=mode, worlds=worlds, decisions=decisions, repeat=repeat,
                   timestep_s=DT, physical_steps_per_world=steps, physics_world_steps=steps * worlds,
                   policy_transitions=decisions * worlds, mean_return=float(episode_return.mean()),
                   returns=episode_return.tolist(), reward_success_worlds=int(reward_fn.succeeded.sum()),
                   scored_success_worlds=sum(s["success"] for s in scores) if scores else None,
                   contact_source="native_mujoco_cpu", policy_device="cpu")
    if trace_dir is not None:
        summary["trace_files"] = files
    elif trace:
        summary["traces"] = embedded
    return {key: torch.stack(values) for key, values in records.items()}, scores, summary

"""Localize discrepancies in the original Panda reach checkpoint audit from frozen inputs."""

import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from _panda_ppo_reach import FEATURE_COUNT, ReachTask, audit_numpy
from panda_state_ppo import (
    _check_fp32_gae_returns, _sha, _write, generalized_advantage,
    load_checkpoint, model_digest, networks,
)


FEATURE_RTOL, FEATURE_ATOL = 2e-4, 2e-5


def feature_components():
    groups = (
        ("hand frame pin translation", .15), ("hand frame pin rotation column 0", None),
        ("hand frame pin rotation column 1", None), ("initial board frame board translation", .03),
        ("initial board frame board rotation column 0", None), ("initial board frame board rotation column 1", None),
        ("world frame pin displacement from reset", .03),
    )
    return [(f"{name} {axis}", scale) for name, scale in groups for axis in "xyz"]


def differences(stored, reference, rtol, atol):
    stored, reference = np.asarray(stored), np.asarray(reference)
    if stored.shape != reference.shape:
        raise ValueError(f"stored shape {stored.shape} differs from reference {reference.shape}")
    if not stored.size or not np.isfinite(stored).all() or not np.isfinite(reference).all():
        raise ValueError("expected finite, nonempty comparison arrays")
    diff = np.abs(stored.astype(np.float64) - reference.astype(np.float64))
    limit = atol + rtol * np.abs(reference.astype(np.float64))
    return diff, limit


def compare(stored, reference, rtol, atol):
    diff, limit = differences(stored, reference, rtol, atol)
    outside = diff > limit
    worst = np.unravel_index((diff - limit).argmax(), diff.shape)
    return dict(count=int(diff.size), outside_count=int(outside.sum()), rtol=rtol, atol=atol,
                max_abs_error=float(diff.max()), largest_excess_index=list(worst),
                stored_at_largest_excess=float(stored[worst]), reference_at_largest_excess=float(reference[worst]),
                error_at_largest_excess=float(diff[worst]), tolerance_at_largest_excess=float(limit[worst]))


def features(stored, native, cpu):
    if (stored.ndim != 3 or stored.shape[-1] != FEATURE_COUNT or cpu.shape != stored.shape
            or not np.isfinite(cpu).all()):
        raise ValueError("expected [decision, world, 21] reach features for both FP32 paths")
    diffs, limits = differences(stored, native, FEATURE_RTOL, FEATURE_ATOL)
    outside = diffs > limits
    components, exceptions = [], []
    for i, (name, scale) in enumerate(feature_components()):
        delta, threshold = diffs[..., i], limits[..., i]
        components.append(dict(feature=i, name=name, stored_unit="dimensionless", meters_per_feature_unit=scale,
                               count=int(delta.size), outside_count=int(outside[..., i].sum()),
                               max_abs_error=float(delta.max()), max_error_over_tolerance=float((delta/threshold).max()),
                               max_abs_error_m=float(delta.max() * scale) if scale is not None else None,
                               max_abs_cpu_fp32_minus_native=float(np.abs(cpu[..., i] - native[..., i]).max()),
                               max_abs_stored_minus_cpu_fp32=float(np.abs(stored[..., i] - cpu[..., i]).max())))
        for t, w in np.argwhere(outside[..., i]):
            k = (t, w, i)
            record = dict(decision=int(t), world=int(w), feature=i, name=name, stored=float(stored[k]),
                          native=float(native[k]), cpu_torch_fp32=float(cpu[k]), abs_error=float(diffs[k]),
                          tolerance=float(limits[k]), error_over_tolerance=float(diffs[k] / limits[k]),
                          cpu_fp32_minus_native=float(cpu[k] - native[k]),
                          stored_minus_cpu_fp32=float(stored[k] - cpu[k]))
            if scale is not None:
                record.update(stored_m=record["stored"] * scale, native_m=record["native"] * scale,
                              abs_error_m=record["abs_error"] * scale, tolerance_m=record["tolerance"] * scale)
            exceptions.append(record)
    return dict(count=int(stored.size), outside_count=len(exceptions), rtol=FEATURE_RTOL, atol=FEATURE_ATOL,
                components=components, exceptions=exceptions)


def boolean_comparison(stored, independent):
    if stored.dtype != bool or independent.dtype != bool or stored.shape != independent.shape:
        raise ValueError("expected boolean masks with matching shapes")
    idx = np.argwhere(stored != independent)
    return dict(count=int(stored.size), mismatch_count=len(idx), mismatch_indices=idx.tolist())


def ppo_replay(checkpoint, data, physical_reward):
    contract = checkpoint["contract"]
    rewards, values, bootstrap = [data[k] for k in ("reward", "value", "bootstrap")]
    saved = data["returns"].reshape(values.shape)
    gamma, lam = contract["gamma"], contract["gae_lambda"]
    try:
        _check_fp32_gae_returns(rewards, values, bootstrap, saved, gamma, lam)
        fp32_returns = dict(exact_match=True, count=int(saved.size))
    except AssertionError as exc:
        fp32_returns = dict(exact_match=False, detail=str(exc))

    def double_replay(reward):
        a, r = generalized_advantage(torch, torch.tensor(reward, dtype=torch.float64),
                                     torch.tensor(values, dtype=torch.float64), torch.tensor(bootstrap, dtype=torch.float64),
                                     gamma, lam)
        a = a.numpy().ravel()
        return (a - a.mean()) / (a.std() + 1e-8), r.numpy()

    saved_norm = data["advantages"]
    norm, recorded_returns = double_replay(rewards)
    physical_norm, physical_returns = double_replay(physical_reward)
    result = dict(recorded_fp32_returns=fp32_returns,
                  normalized_recorded_advantages=compare(saved_norm, norm, 5e-5, 5e-5),
                  normalized_physical_advantages=compare(saved_norm, physical_norm, 5e-5, 5e-5))
    if fp32_returns["exact_match"]:
        # Include the original audit's independently propagated reward discrepancy and FP32 rounding.
        error = np.abs(rewards - physical_reward)
        for t in range(len(error) - 2, -1, -1):
            error[t] += gamma * lam * error[t+1]
        error += np.abs(saved - recorded_returns) + 3e-5 + 3e-5 * np.abs(recorded_returns)
        result["physical_returns"] = dict(count=int(saved.size),
                                          outside_count=int((np.abs(saved - physical_returns) > error).sum()))
    probe = checkpoint["action_probe"]
    result["actions"] = {}
    with torch.no_grad():
        for which in ("before", "after"):
            actor, _ = networks(torch, "cpu", contract["observation_size"])
            actor.load_state_dict(probe[which + "_actor"])
            mean, log_std = actor(probe["observation"])
            action = (mean + log_std.exp() * probe["noise"]).tanh()
            # torch.allclose in the original audit takes the stored action as its reference.
            device_action = probe[which + "_action"]
            result["actions"][which] = dict(original_torch_allclose=torch.allclose(
                action, device_action, rtol=2e-5, atol=2e-5),
                max_abs_error=float((action - device_action).abs().max()))
    result["post_update_actor_equal"] = all(torch.equal(weight, probe["after_actor"][k])
                                            for k, weight in checkpoint["actor"].items())
    metrics = data["optimizer_metrics"]
    result["finite_optimizer_metrics_with_nonzero_gradient"] = bool(np.isfinite(metrics).all()
                                                                    and (metrics[:, 3] > 0).any())
    return result


def diagnose(path, rollout=None, scene=None):
    checkpoint = load_checkpoint(torch, path)
    contract = checkpoint["contract"]
    if contract.get("task") != "reach":
        raise ValueError("expected a physical reach checkpoint")
    rollout = rollout or path.parent / checkpoint["rollout_file"]
    scene = scene or path.parent / "scene.xml"
    rollout_hash, scene_hash = _sha(rollout), _sha(scene)
    if rollout_hash != checkpoint["rollout_sha256"] or scene_hash != contract["model_sha256"]:
        raise ValueError("rollout or scene SHA256 differs from checkpoint binding")
    model = mj.MjModel.from_xml_path(str(scene))
    compiled_hash = model_digest(model)
    if compiled_hash != checkpoint["frozen_model_sha256"]:
        raise ValueError("compiled model differs from checkpoint binding")
    task = ReachTask(torch, model, "cpu")
    if task.contract != contract["reach"]:
        raise ValueError("compiled reach FK or feature contract differs from checkpoint")

    with np.load(rollout, allow_pickle=False) as data:
        q, v, u = (data[k] for k in ("next_qpos", "next_qvel", "controls"))
        reset, limit = data["reset_qpos"], data["effort_limit"]
        native = audit_numpy(model, reset, q, v, u, limit)
        with torch.no_grad():
            task.reset(torch.from_numpy(reset))
            cpu = np.stack([task.measure(torch.from_numpy(row))["features"].numpy() for row in q])
        feature_report = features(data["reach_features"], native["features"], cpu)
        metrics = {k: compare(data["reach_" + k], values, FEATURE_RTOL, FEATURE_ATOL)
                   for k, values in native.items() if k not in ("features", "qualified", "reward")}
        metrics["qualified"] = boolean_comparison(data["reach_qualified"], native["qualified"])
        metrics["reward"] = compare(data["reward"], native["reward"], 2e-5, 2e-5)
        qs, vs = data["substep_qpos"], data["substep_qvel"]
        t, repeat, w, nq = qs.shape
        if (nq != 23 or vs.shape != (t, repeat, w, 22) or u.shape != (t, repeat, w, 16)
                or q.shape != (t, w, 23) or not np.array_equal(qs[:, -1], q) or not np.array_equal(vs[:, -1], v)):
            raise ValueError("inconsistent saved physics substeps and accepted successors")
        sub = audit_numpy(model, reset, qs.reshape(t*repeat, w, 23), vs.reshape(t*repeat, w, 22),
                          u.reshape(t*repeat, 1, w, 16), limit)
        metrics["substep_qualified"] = boolean_comparison(data["substep_qualified"], sub["qualified"].reshape(t, repeat, w))
        ppo = ppo_replay(checkpoint, data, native["reward"])
    return dict(schema="panda-reach-feature-error-v1", checkpoint_sha256=_sha(path), rollout_sha256=rollout_hash,
                scene_sha256=scene_hash, frozen_model_sha256=compiled_hash, fk_geometry_sha256=task.contract["fk_geometry_sha256"],
                reference="Native MuJoCo FP64 FK of saved physical successor qpos; reset from the same rollout.",
                cpu_comparator="The training FK implementation evaluated on CPU Torch FP32 at the same recorded qpos.",
                decision_indexing="Zero-based successor index; index 0 follows the first decision's physics substeps.",
                features=feature_report, other_physical_checks=metrics, ppo_checks=ppo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, help="defaults to the checkpoint-bound rollout beside the checkpoint")
    parser.add_argument("--scene", type=Path, help="defaults to scene.xml beside the checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.checkpoint, args.rollout, args.scene)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write(args.out, report)
    print(f"{report['features']['outside_count']} of {report['features']['count']} reach feature values "
          f"exceed the original audit tolerance. Details in {args.out}")


if __name__ == "__main__":
    main()

"""Imitate a reach controller on the original S4000 PPO training reset bank."""
import argparse
import math
from pathlib import Path
from time import perf_counter

import numpy as np

import panda_state_ppo as ppo


SEED_SCHEMA = "panda-reach-imitation-seed-v1"


def _verify_resident(begin, end, steps, worlds, bias):
    counts = {k: end[k] - begin[k] for k in (
        "resident_steps", "accepted_world_steps", "failed_batches", "explicit_readback_bytes",
        "downloaded_bytes", "status_downloaded_bytes", "torch_to_tyx_bytes", "tyx_to_torch_bytes",
        "launched_kernels", "completed_kernels")}
    # One reset-state read, then q/v and (when available) entry bias at each substep.
    minimum_state_bytes = 4 * worlds * ((steps + 1) * (23 + 22) + (22 * steps if bias == "lagged" else 0))
    expected = dict(resident_steps=steps, accepted_world_steps=worlds * steps, failed_batches=0,
                    explicit_readback_bytes=0, downloaded_bytes=4 * worlds * steps,
                    status_downloaded_bytes=4 * worlds * steps, torch_to_tyx_bytes=4 * worlds * 16 * steps)
    if (any(counts[k] != v for k, v in expected.items())
            or counts["tyx_to_torch_bytes"] < minimum_state_bytes
            or counts["launched_kernels"] <= 0 or counts["launched_kernels"] != counts["completed_kernels"]):
        raise RuntimeError(f"Unexpected resident physics or transport counts: {counts}")
    return counts


def rollout(torch, task, actor, targets, steps, expert_weight, reset_q, reset_v, *, teacher_path="direct",
            collect_labels=True, track_pin_excursion=False):
    """A deterministic policy diagnostic omits the teacher and produces no labels."""
    from _panda_ppo_reach import DWELL_S

    begin = task.bank.report()
    started = perf_counter()
    observations, labels = [], []
    stage = None
    with torch.no_grad():
        obs = task.reset()
        if targets is not None:
            if teacher_path in ("axial", "axial-recovery"):
                if teacher_path == "axial-recovery":
                    from _panda_reach_teacher import recoverable_waypoint_action as waypoint_action
                else:
                    from _panda_reach_teacher import waypoint_action
                stage = torch.zeros(task.worlds, dtype=torch.int64, device=task.device)
            else:
                from _panda_reach_teacher import teacher_action
        if not torch.equal(task.q, reset_q) or not torch.equal(task.v, reset_v):
            raise ValueError("Resident reset did not reproduce the original training qpos/qvel")
        first_distance = task.reach_measurement["distance_m"].clone()
        nearest = first_distance.clone()
        if track_pin_excursion:
            peak_drift = task.reach_measurement["part_drift_m"].clone()
        returns = torch.zeros_like(nearest)
        actions_valid = torch.ones((), dtype=torch.bool, device=task.device)
        for _ in range(steps):
            if targets is None:
                action = actor(obs)[0].tanh()
            else:
                label = (teacher_action(torch, task, targets) if stage is None
                         else waypoint_action(torch, task, targets, stage))
                if collect_labels:
                    observations.append(obs)
                    labels.append(label.clone())
                if expert_weight == 1.:
                    action = label
                else:
                    policy = actor(obs)[0].tanh()
                    action = policy if expert_weight == 0. else torch.lerp(policy, label, expert_weight)
                actions_valid.logical_and_((label.abs() <= 1.).all())
            actions_valid.logical_and_((action.abs() <= 1.).all())
            obs, reward, _ = task.step(action)
            returns += reward
            nearest = torch.minimum(nearest, task.reach_measurement["distance_m"])
            if track_pin_excursion:
                # Evaluate the same pin grasp site used by the reach contract at
                # each accepted substep, including excursions between decisions.
                pin, _ = task.reach.poses(task.substep_qpos.flatten(0, 1))[2]
                delta = pin.reshape(-1, task.worlds, 3) - task.reach.reference[2]
                peak_drift = torch.maximum(peak_drift, delta.square().sum(-1).sqrt().amax(0))
        if not bool((actions_valid & task.finite).cpu()):
            raise RuntimeError("Nonfinite rollout or action outside [-1, 1]")
        gate = task.reach_max_dwell_steps >= math.ceil(DWELL_S / task.bank.timestep)
        count = int(gate.sum().cpu())
        dwell = task.reach_max_dwell_steps.cpu().tolist()
        measurement = {k: x.cpu().tolist() for k, x in task.reach_measurement.items() if k != "features"}
    end = task.bank.report()
    metrics = dict(expert_weight=expert_weight, worlds=task.worlds, qualified_worlds=count,
                   qualified_rate=count / task.worlds, qualified_by_world=gate.cpu().tolist(),
                   max_dwell_steps_by_world=dwell, initial_distance_m=first_distance.cpu().tolist(),
                   closest_decision_distance_m=nearest.cpu().tolist(), final=measurement,
                   return_by_world=returns.cpu().tolist(),
                   physics=_verify_resident(begin, end, steps * task.args.action_repeat, task.worlds, task.args.bias),
                   collection_seconds=perf_counter() - started)
    if stage is not None:
        metrics["final_waypoint_stage_by_world"] = stage.cpu().tolist()
    if track_pin_excursion:
        metrics["peak_pin_displacement_m_by_world"] = peak_drift.cpu().tolist()
    data = (torch.cat(observations), torch.cat(labels)) if labels else None
    return data, metrics


def physical_score(metrics, pin_limit):
    if metrics["mode"] not in ("learner_labeled", "learner_final", "parent") or metrics["expert_weight"] != 0.:
        raise ValueError("Actor selection requires a zero-teacher-action policy rollout")
    final = metrics["final"]
    peak = np.asarray(metrics["peak_pin_displacement_m_by_world"])
    distance = np.asarray(final["distance_m"])
    if not np.isfinite(peak).all() or not np.isfinite(distance).all():
        raise ValueError("Nonfinite physical candidate measurement")
    displaced = int((peak > pin_limit).sum())
    dwell = metrics["qualified_worlds"]
    endpoint = sum(final["qualified"])
    mean_distance = float(distance.mean())
    return dict(mode=metrics["mode"], expert_weight=metrics["expert_weight"], pin_excursion_worlds=displaced,
                final_displaced_pin_worlds=int((np.asarray(final["part_drift_m"]) > pin_limit).sum()),
                dwell_qualified_worlds=dwell, final_qualified_worlds=endpoint, mean_final_distance_m=mean_distance,
                ascending_score=[displaced, -dwell, -endpoint, mean_distance])


def fit(torch, actor, optimizer, data, epochs, batch):
    # Weight by the observed size of each motor command so that smaller
    # corrective joint movements receive gradient alongside the approach.
    obs = torch.cat([x for x, _ in data])
    labels = torch.cat([y for _, y in data])
    scale = labels.square().mean(dim=0).sqrt().clamp(min=.10)
    # A bounded pre-tanh target supplies gradients even if the parent mean
    # saturates a motor. A small residual in action space matters near zero.
    raw_target = torch.atanh(labels.clamp(-.995, .995))
    torch.musa.synchronize()
    started = perf_counter()
    with torch.no_grad():
        previous = (actor(obs)[0].tanh() - labels).square().mean()
    for _ in range(epochs):
        for ids in torch.randperm(len(obs), device=obs.device).split(batch):
            raw, _ = actor(obs[ids])
            error = (raw.tanh() - labels[ids]) / scale
            raw_error = (raw - raw_target[ids]) / scale
            loss = error.square().mean() + .05 * raw_error.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.body.parameters(), 1.)
            optimizer.step()
    with torch.no_grad():
        after = (actor(obs)[0].tanh() - labels).square().mean()
        if not bool(torch.isfinite(after).cpu()):
            raise RuntimeError("Nonfinite imitation fit")
    return dict(samples=len(obs), mse_before=float(previous.cpu()), mse_after=float(after.cpu()),
                epochs=epochs, fit_seconds=perf_counter() - started)


def extended_training_bank(train, scene, model, original_arrays, reset_sha, out):
    from panda_reach_crossed_resets import generate_bank

    extra_dir = out / "train-extra"
    q, v, extra = generate_bank(scene, model, "train-extra", extra_dir)
    qpos, qvel = (np.concatenate((original_arrays[k], new), axis=0)
                  for k, new in (("qpos", q), ("qvel", v)))
    if any(np.all(original_arrays["qpos"] == row, axis=1).any() for row in q):
        raise ValueError("A generated train-extra pose duplicates a parent reset")
    path = out / "demonstration-states.npz"
    np.savez_compressed(path, qpos=qpos, qvel=qvel, model_sha256=scene.xml_sha256)
    # The immutable 12-world PPO contract remains bound to the parent. The
    # larger namespace only configures the resident demonstration session.
    demonstration = argparse.Namespace(**vars(train))
    demonstration.worlds, demonstration.reset_npz = len(qpos), path
    initial, digest, arrays = ppo.physical_initial_states(scene, demonstration)
    np.testing.assert_array_equal(arrays["qpos"], qpos)
    np.testing.assert_array_equal(arrays["qvel"], qvel)
    receipt = dict(
        schema="panda-reach-extended-demonstration-bank-v1", ppo_worlds=train.worlds,
        demonstration_worlds=demonstration.worlds, model_sha256=scene.xml_sha256,
        physical_reset_sha256=digest, file=path.name, file_sha256=ppo._sha(path),
        parent=dict(worlds=train.worlds, file=str(train.reset_npz.resolve()),
                    file_sha256=ppo._sha(train.reset_npz), physical_reset_sha256=reset_sha),
        extra=dict(split=extra["split"], worlds=len(q), seed=extra["seed"], levels=extra["levels"],
                   generator_sha256=ppo._sha(Path(__file__).with_name("panda_reach_crossed_resets.py")),
                   file="train-extra/initial-states.npz", file_sha256=extra["file_sha256"],
                   metadata_file="train-extra/metadata.json", metadata_sha256=ppo._sha(extra_dir / "metadata.json"),
                   physical_reset_sha256=extra["physical_reset_sha256"], qpos=q.tolist(), qvel=v.tolist()),
        world_mapping=[dict(world=i, bank="parent", bank_row=i) for i in range(train.worlds)] +
                      [dict(world=train.worlds + row["world"], bank="train-extra", bank_row=row["world"],
                            cell=row["cell"], replicate=row["replicate"]) for row in extra["worlds"]])
    return demonstration, initial, arrays, receipt


def run(args):
    from _panda_grasp_insert_scene import build_scene
    from _panda_ppo_reach import ReachTask
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime

    torch = ppo.require_musa()
    torch.musa.set_device(0)
    name = torch.musa.get_device_name(0)
    if "S4000" not in name:
        raise RuntimeError(f"S4000 required; found {name}")
    device = torch.device("musa:0")
    source_sha = ppo._sha(args.checkpoint)
    parent = ppo.load_checkpoint(torch, args.checkpoint)
    original = parent["contract"]
    if (original["task"] != "reach" or type(parent["completed_update"]) is not int
            or parent["completed_update"] < 0 or parent["next_update"] != parent["completed_update"] + 1):
        raise ValueError("Expected a completed reach PPO update")
    train = argparse.Namespace(**original)
    train.stage, train.reset_npz = "train", args.reset_npz
    if args.include_crossed_training and train.worlds != 12:
        raise ValueError("Extended demonstrations require the original 12-world parent contract")
    training_worlds = 24 if args.include_crossed_training else train.worlds
    if not 1 <= args.minimum_teacher_worlds <= training_worlds:
        raise ValueError(f"--minimum-teacher-worlds must be between 1 and {training_worlds}")
    if train.observation_size != 98 or train.action_size != 16 or train.action_repeat != 8:
        raise ValueError("Expected the original 98D reach input, 16 motors, and eight physics substeps")
    torch.manual_seed(args.seed)
    torch.musa.manual_seed_all(args.seed)
    scene = build_scene(urdf_path=args.urdf)
    source = scene.compile()
    model = source.mj_model
    frozen = ppo.model_digest(model)
    initial, reset_sha, arrays = ppo.physical_initial_states(scene, train)
    if reset_sha != original["physical_reset_sha256"]:
        raise ValueError("Only the parent checkpoint's physical training reset bank is permitted for imitation")
    reach = ReachTask(torch, model, device)
    contract = ppo.config(train, scene)
    contract.update(physical_reset_sha256=reset_sha, reach=reach.contract,
                    reach_source_sha256=ppo._sha(Path(__file__).with_name("_panda_ppo_reach.py")))
    if contract != original or frozen != parent["frozen_model_sha256"]:
        raise ValueError("Physical model, observation, action, or PPO contract differs from the parent checkpoint")
    extended = None
    if args.include_crossed_training:
        args.out.mkdir(parents=True, exist_ok=False)
        train, initial, arrays, extended = extended_training_bank(train, scene, model, arrays, reset_sha, args.out)
    if args.teacher_path in ("axial", "axial-recovery"):
        from _panda_reach_teacher import build_waypoints
        targets_np = np.asarray(build_waypoints(model, arrays["qpos"].copy()))
        target_shape = (train.worlds, 4, 16)
    else:
        from _panda_reach_teacher import build_targets
        targets_np = np.asarray(build_targets(model, arrays["qpos"].copy()))
        target_shape = (train.worlds, 16)
    if targets_np.shape != target_shape or not np.isfinite(targets_np).all():
        raise ValueError(f"Expected finite {args.teacher_path} targets of shape {target_shape}")
    targets = torch.as_tensor(targets_np, device=device, dtype=torch.float32)
    reset_q, reset_v = (torch.as_tensor(arrays[k], device=device, dtype=torch.float32) for k in ("qpos", "qvel"))
    actor, _ = ppo.networks(torch, device, train.observation_size)
    actor.load_state_dict(parent["actor"])
    # Only the deterministic mean network receives imitation gradients.
    optimizer = torch.optim.Adam(actor.body.parameters(), lr=args.fit_lr)
    if extended is None:
        args.out.mkdir(parents=True, exist_ok=False)
    summary = dict(status="running", device=name, parent_checkpoint_sha256=source_sha, source_update=parent["completed_update"],
                   physical_reset_sha256=reset_sha, reset_npz_sha256=ppo._sha(args.reset_npz),
                   frozen_model_sha256=frozen, contract=contract, seed=args.seed,
                   teacher_sha256=ppo._sha(Path(__file__).with_name("_panda_reach_teacher.py")),
                   script_sha256=ppo._sha(__file__), rollout_steps=args.steps, fit_lr=args.fit_lr,
                   teacher_path=args.teacher_path, minimum_teacher_worlds=args.minimum_teacher_worlds,
                   teacher_only=args.teacher_only, select_physical_seed=args.select_physical_seed,
                   fit_epochs=args.fit_epochs, minibatch=args.batch, rollouts=[], fits=[],
                   scope="Physical reach diagnostics on the original PPO training reset bank")
    data = []
    candidates = {}
    modes = [("teacher", 1.), ("mixed", .35), ("learner_labeled", 0.), ("learner_final", 0.)]
    if extended is not None:
        summary.update(scope="Physical reach diagnostics on the parent and generated train-extra resets",
                       demonstration_training=extended)
    if args.select_physical_seed:
        summary["physical_selection"] = dict(
            reset_scope="original physical training reset bank", pin_limit_m=reach.contract["success"]["max_pin_displacement_m"],
            rule=["fewest worlds exceeding pin displacement limit at any accepted physics substep",
                  "most worlds reaching qualified dwell", "most worlds jointly qualified at the final state",
                  "lowest mean final hand-to-pin distance", "prefer learner_final on exact tie"], candidates={})
        if extended is not None:
            selection = summary["physical_selection"]
            selection.update(reset_scope=summary["scope"], physical_reset_sha256=extended["physical_reset_sha256"])
            selection["rule"][-1] = "retain parent on exact tie, then prefer learner_final among learners"
            modes.append(("parent", 0.))
    try:
        with MuJoCoRuntime(source, mode="musa", world_keys=tuple(range(train.worlds)), initial_states=initial,
                           cache_root=args.cache, contact_capacity=64, constraint_capacity=256) as runtime:
            with runtime.musa_resident() as bank:
                task = ppo.DeviceTask(torch, device, bank, model, scene.initial_qpos, train, reach=reach)
                for mode, weight in modes:
                    if mode == "parent":
                        actor.load_state_dict(parent["actor"])
                    training = mode in ("teacher", "mixed", "learner_labeled")
                    candidate = args.select_physical_seed and mode in ("learner_labeled", "learner_final", "parent")
                    batch, metrics = rollout(torch, task, actor, targets if training else None,
                                             args.steps, weight, reset_q, reset_v, teacher_path=args.teacher_path,
                                             collect_labels=training and not args.teacher_only,
                                             track_pin_excursion=candidate)
                    metrics["mode"] = mode
                    summary["rollouts"].append(metrics)
                    if candidate:
                        # Capture the deployed weights before the labeled pass is fitted.
                        candidates[mode] = ppo._cpu_state(actor)
                        selection = summary["physical_selection"]
                        selection["candidates"][mode] = physical_score(metrics, selection["pin_limit_m"])
                    ppo._write(args.out / "progress.json", summary)
                    print(f"{mode} reach {metrics['qualified_worlds']}/{train.worlds}, "
                          f"max dwell steps {metrics['max_dwell_steps_by_world']}, "
                          f"final distance (m) {metrics['final']['distance_m']}", flush=True)
                    if mode == "teacher":
                        count = metrics["qualified_worlds"]
                        if count < args.minimum_teacher_worlds:
                            raise RuntimeError(f"{args.teacher_path} teacher met the reach dwell condition in "
                                               f"{count}/{train.worlds} training worlds; "
                                               f"minimum qualified worlds is {args.minimum_teacher_worlds}")
                        if args.teacher_only:
                            break
                    if training:
                        data.append(batch)
                        summary["fits"].append(fit(torch, actor, optimizer, data, args.fit_epochs, args.batch))
                        ppo._write(args.out / "progress.json", summary)
        if ppo.model_digest(model) != frozen:
            raise RuntimeError("Compiled physical model changed during imitation")
        if args.teacher_only:
            summary["status"] = "completed"
            return summary
        if args.select_physical_seed:
            selection = summary["physical_selection"]
            scores = selection["candidates"]
            order = ("learner_final", "learner_labeled") if extended is None else ("parent", "learner_final", "learner_labeled")
            chosen = min(order, key=lambda mode: scores[mode]["ascending_score"])
            selection.update(selected_mode=chosen, selected_score=scores[chosen]["ascending_score"])
            if chosen == "parent":
                summary.update(status="completed", actor_seed=None,
                               promotion="parent retained; neither learner strictly improved the physical score")
                return summary
            actor_state = candidates[chosen]
        else:
            actor_state = ppo._cpu_state(actor)
        seed = dict(schema=SEED_SCHEMA, actor=actor_state, contract=original,
                    frozen_model_sha256=frozen, parent_checkpoint_sha256=source_sha)
        if extended is not None:
            seed["demonstration_training"] = extended
            if args.select_physical_seed:
                seed["physical_selection"] = summary["physical_selection"]
        ppo.validate_checkpoint_data(torch, seed)
        path = args.out / "actor-seed.pt"
        torch.save(seed, path)
        summary.update(status="completed", actor_seed=path.name, actor_seed_sha256=ppo._sha(path))
    except BaseException as exc:
        summary.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        ppo._write(args.out / "summary.json", summary)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--reset-npz", type=Path, required=True, help="original physical training reset bank")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--urdf", type=Path)
    p.add_argument("--include-crossed-training", action="store_true",
                   help="include twelve generated train-extra resets during imitation and policy selection")
    p.add_argument("--teacher-path", choices=("direct", "axial", "axial-recovery"), default="direct")
    p.add_argument("--select-physical-seed", action="store_true",
                   help="choose between the two zero-teacher-weight actors on the original training bank")
    p.add_argument("--minimum-teacher-worlds", type=int, default=1,
                   help="require this many training worlds to meet the physical dwell condition before fitting")
    p.add_argument("--teacher-only", action="store_true", help="run only the teacher's physical diagnostic")
    p.add_argument("--steps", type=int, default=128, help="physical decisions per rollout")
    p.add_argument("--fit-epochs", type=int, default=100, help="aggregate offline passes after each labeled rollout")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--fit-lr", type=float, default=.001)
    p.add_argument("--seed", type=int, default=91125)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if min(args.steps, args.fit_epochs, args.batch) <= 0 or not math.isfinite(args.fit_lr) or args.fit_lr <= 0 or args.seed < 0:
        raise SystemExit("Positive rollout and fit sizes, finite positive learning rate, and nonnegative seed required")
    run(args)

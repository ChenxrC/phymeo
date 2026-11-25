"""Episode-return cross-entropy training and paired frozen evaluation on dual wrist RGB."""

import argparse
import hashlib
import json
import os
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from _bimanual_visual_policy import PARAMETERS, PRIOR, VisualPolicy


HOLDOUT = (43001, 43002, 43003, 43004)
SELECTION = (32001, 32002)


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _bytes(value):
    return json.dumps(value, default=_json, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _sha(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        name = tmp.name
        try:
            tmp.write(_bytes(value))
            tmp.flush()
            os.fsync(tmp.fileno())
        except BaseException:
            tmp.close()
            Path(name).unlink(missing_ok=True)
            raise
    try:
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _read(path):
    return json.loads(path.read_bytes())


def _dofs(seed):
    return (6, 7) if seed % 2 else (7, 6)


def _job(label, weights, arms="both", black=False):
    return dict(label=label, weights=np.asarray(weights, float).tolist(), arms=arms, black=black)


def _sample_population(mean, std, rng, n):
    noise = rng.normal(size=((n + 1) // 2, len(mean)))
    # Retain the current mean. Reflected deviations reduce episode sampling variance.
    deviations = np.concatenate((noise, -noise), axis=0)[:n - 1]
    return np.clip(np.vstack((mean, mean + std * deviations)), -1.5, 1.5)


def _population_update(population, scores, elite_count, fraction):
    if not np.isfinite(scores).all():
        raise ValueError("CEM requires completed episodes with finite returns")
    order = np.argsort(-np.asarray(scores), kind="stable")
    elite = population[order[:elite_count]]
    return (1 - fraction) * population[0] + fraction * elite.mean(axis=0), elite.std(axis=0), int(order[0])


class Runner:
    def __init__(self, args):
        self.args = args
        self.begin = time.monotonic()

    def check_time(self):
        if self.args.wall_seconds and time.monotonic() - self.begin >= self.args.wall_seconds:
            raise TimeoutError("wall budget reached; completed episodes and generation checkpoint retained")

    def episodes(self, stage, seed, jobs, *, keep_trajectory=False):
        """Every finished lane has its own durable receipt and can be resumed independently."""
        results, pending = {}, []
        for job in jobs:
            path = self.args.out / "episodes" / stage / f"{seed}-{job['label']}.json"
            if path.is_file():
                row = _read(path)
                if row["job"] != job or row["seed"] != seed:
                    raise ValueError(f"saved episode conflicts with requested policy: {path}")
                results[job["label"]] = row
            else:
                pending.append((job, path))
        for offset in range(0, len(pending), self.args.batch):
            self.check_time()
            group = pending[offset:offset + self.args.batch]
            results.update(self._batch(seed, group, keep_trajectory))
        return [results[j["label"]] for j in jobs]

    def _batch(self, seed, group, keep):
        from bimanual_musa_env import BimanualBatchEnv
        import bimanual_wrist_render as wrist_render
        from bimanual_wrist_render import BimanualWristRenderer

        args = self.args
        fiducials = args.policy_variant == "fiducial"
        keys = tuple(range(len(group)))
        cache = args.cache_dir
        with ExitStack() as stack:
            env = stack.enter_context(BimanualBatchEnv(args.task, seed=seed, world_keys=keys,
                                      arm_dofs=_dofs(seed), mode=args.mode, cache_root=cache,
                                      action_repeat=args.action_repeat, horizon=args.horizon,
                                      tool_profile=args.tool_profile, load_profile=args.load_profile, stance=args.stance))
            packets = env.reset(seed=seed, active_arms={k: group[k][0]["arms"] for k in keys})
            scene = env.scene
            visual = stack.enter_context(BimanualWristRenderer(env.model, cache / "camera",
                                         library_path=args.library, size=(args.image_size,) * 2,
                                         device_index=args.device_index, fiducials=fiducials))
            actors = [VisualPolicy(args.task, env.model, job["weights"], variant=args.policy_variant,
                                   decision_dt=args.action_repeat*scene.timestep,
                                   servo_gains=(env.worlds[k].kp, env.worlds[k].kd))
                      for k, (job, _) in enumerate(group)]
            digests = [hashlib.sha256() for _ in keys]
            observations = [p[0] for p in (packets[k] for k in keys)]
            infos = [packets[k][1] for k in keys]
            histories = [[(np.array(info["qpos"], copy=True), int(info["tick"]), float(info["time_s"]))] for info in infos]
            actions_recorded = [[] for _ in keys]
            last_features = [{} for _ in keys]
            diagnostics = [[] for _ in keys]
            visible = np.zeros((len(keys), 2, 2), int)
            active = set(keys)
            rewards = [[] for _ in keys]
            final = {}
            device_start = env.runtime.musa_device_report() if args.mode == "musa" else None
            for decision in range(args.horizon):
                controls = {k: np.zeros(14) for k in keys}
                for k in sorted(active):
                    info = infos[k]
                    image = visual.observe(info["qpos"], info["tick"], info["time_s"], epoch=k)
                    digests[k].update(np.ascontiguousarray(image).tobytes())
                    actor_input = np.zeros_like(image) if group[k][0]["black"] else image
                    if fiducials:
                        controls[k] = actors[k].act(observations[k], actor_input, wrist_force_world_n=info["wrist_force_world_n"])
                    else:
                        controls[k] = actors[k].act(observations[k], actor_input)
                    actions_recorded[k].append(controls[k].copy())
                    last_features[k] = actors[k].features
                    visible[k] += np.asarray(last_features[k]["pixel_counts"]) >= 4
                post = env.step_torque(controls) if fiducials else env.step(controls)
                for k in sorted(active.copy()):
                    observations[k], reward, term, trunc, info = post[k]
                    infos[k] = info
                    reward = float(reward)
                    if not np.isfinite(reward):
                        raise ValueError("nonfinite environment reward")
                    rewards[k].append(reward)
                    if keep:
                        diagnostics[k].append(dict(input_tick=histories[k][-1][1], output_tick=int(info["tick"]),
                                                   visual=last_features[k], measured={name: info[name] for name in (
                                                       "payload_xyz_m", "payload_yaw_rad", "clearance_m", "goal_error_xy_m",
                                                       "goal_yaw_error_rad", "pad_normal_n", "pad_vertical_n",
                                                       "floor_vertical_n", "both_loaded_airborne_s", "carried_airborne",
                                                       "goal_dwell_s")}))
                    histories[k].append((np.array(info["qpos"], copy=True), int(info["tick"]), float(info["time_s"])))
                    if term or trunc:
                        final[k] = dict(terminated=bool(term), truncated=bool(trunc), success=bool(info["success"]))
                        active.remove(k)
                if not active:
                    break
            if active:
                raise RuntimeError("environment did not finish its declared decision horizon")
            device_end = env.runtime.musa_device_report() if args.mode == "musa" else None
            if args.mode == "musa":
                if not (device_end["backend"] == "musa" and device_end["arch"] and device_end["device_name"]
                        and device_end["completed_kernels"] > device_start["completed_kernels"]
                        and device_end["completed_kernels"] == device_end["launched_kernels"]):
                    raise RuntimeError("missing physical MUSA launch/completion evidence")
            rows = {}
            for k, (job, path) in enumerate(group):
                path.parent.mkdir(parents=True, exist_ok=True)
                if keep:
                    xml = path.with_suffix(".xml")
                    xml.write_bytes(scene.xml.encode())
                    q, ticks, times = zip(*histories[k], strict=True)
                    np.savez_compressed(path.with_suffix(".npz"), qpos=q, ticks=ticks, time_s=times,
                                        model_sha256=scene.xml_sha256, action=actions_recorded[k], rewards=rewards[k])
                indicators = {name: val for name, val in infos[k].items()
                              if name not in ("qpos", "qvel", "applied_ctrl")}
                row = dict(job=job, seed=seed, arm_dofs=_dofs(seed), task=args.task, physics=args.mode,
                           fiducials=fiducials, tool_profile=args.tool_profile, load_profile=args.load_profile,
                           stance=args.stance, marker_layout_version=(
                               getattr(wrist_render, "MARKER_LAYOUT_VERSION", None) if fiducials else None),
                           action_semantics="bounded_motor_torque_fraction" if fiducials else "bounded_joint_increment",
                           return_sum=float(np.sum(rewards[k])), decisions=len(rewards[k]), rewards=rewards[k],
                           **final[k], scene_sha256=scene.xml_sha256, sensor_rgb_sha256=digests[k].hexdigest(),
                           last_features=last_features[k], image_visible_decisions=visible[k], final_metrics=indicators,
                           device_before=device_start, device_after=device_end, renderer=visual.device_report,
                           batch_size=len(keys), physics_ticks=histories[k][-1][1])
                if keep:
                    row["decision_diagnostics"] = diagnostics[k]
                    row["trajectory_sha256"] = hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest()
                _save(path, row)
                rows[job["label"]] = row
                print(f"{args.task} {seed} {job['label']} {args.mode} return={row['return_sum']:.4f} "
                      f"success={row['success']} ticks={row['physics_ticks']}", flush=True)
            return rows


def _contract(args):
    base = Path(__file__).resolve().parent
    files = ("bimanual_musa_train.py", "_bimanual_visual_policy.py", "_bimanual_image_pose.py",
             "bimanual_musa_env.py", "bimanual_wrist_render.py")
    if args.policy_variant == "fiducial":
        files += ("_bimanual_fiducial.py",)
    return dict(task=args.task, arms=args.arms, mode=args.mode, population=args.population, seed=args.seed,
                generations=args.generations, train_seeds=args.train_seeds, horizon=args.horizon,
                policy_variant=args.policy_variant, tool_profile=args.tool_profile, load_profile=args.load_profile,
                stance=args.stance,
                action_repeat=args.action_repeat, image_size=args.image_size, parameters=PARAMETERS,
                selection_seeds=SELECTION, holdout_seeds=HOLDOUT,
                sources={n: hashlib.sha256((base / n).read_bytes()).hexdigest() for n in files})


def _train(args, runner, contract, state):
    for generation in range(state["generation"], args.generations):
        weights = np.array(state["mean"])
        spread = np.array(state["std"])
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, generation]))
        population = _sample_population(weights, spread, rng, args.population)
        jobs = [_job(f"candidate-{i}", p, args.arms) for i, p in enumerate(population)]
        scores = np.zeros(len(jobs))
        seeds = [1001 + generation * args.train_seeds + i for i in range(args.train_seeds)]
        if max(seeds) >= min(SELECTION):
            raise ValueError("training seed range is exhausted")
        for seed in seeds:
            rows = runner.episodes(f"train-{generation:04d}", seed, jobs)
            scores += np.array([r["return_sum"] for r in rows]) / len(seeds)
        mean, elite_std, best = _population_update(population, scores, max(2, len(jobs) // 3), .75)
        std = np.maximum(.10, .35 * spread + .65 * elite_std)
        state["updates"].append(dict(generation=generation, seeds=seeds, scores=scores.tolist(),
                                     mean_before=weights.tolist(), mean_after=mean.tolist(), best=population[best].tolist()))
        state.update(mean=mean.tolist(), std=std.tolist(), generation=generation + 1)
        _save(args.out / "checkpoint.json", state)
    # Select from a small, predeclared shortlist on two unseen development models.
    shortlist = [_job("mean", state["mean"], args.arms), _job("latest-elite", state["updates"][-1]["best"], args.arms)]
    returns = np.array([[r["return_sum"] for r in runner.episodes("selection", seed, shortlist)] for seed in SELECTION])
    winner = int(np.argmax(returns.mean(axis=0)))
    frozen = dict(schema="bimanual-visual-policy-v1", task=args.task, arms=args.arms, variant=args.policy_variant,
                  parameters=PARAMETERS, weights=shortlist[winner]["weights"], selected=shortlist[winner]["label"],
                  tool_profile=args.tool_profile, load_profile=args.load_profile, stance=args.stance,
                  selection_returns=returns, contract_sha256=_sha(contract), generations=state["generation"])
    _save(args.out / "frozen-policy.json", frozen)


def _evaluate(args, runner, policy):
    jobs = [_job("trained", policy["weights"], args.arms), _job("initial-prior", PRIOR, args.arms),
            _job("black-rgb", policy["weights"], args.arms, black=True)]
    if args.arms == "both":
        jobs += [_job("left-only", policy["weights"], "left"), _job("right-only", policy["weights"], "right")]
    episodes = [row for seed in HOLDOUT for row in runner.episodes("holdout", seed, jobs, keep_trajectory=True)]
    report = {}
    for job in jobs:
        rows = [r for r in episodes if r["job"]["label"] == job["label"]]
        successes = [r for r in rows if r["success"]]
        represented = {tuple(r["arm_dofs"]) for r in successes}
        report[job["label"]] = dict(seeds=[r["seed"] for r in rows], success=[r["success"] for r in rows],
                                    returns=[r["return_sum"] for r in rows], count=len(successes),
                                    meets_gate=len(successes) >= 3 and len(represented) == 2)
    _save(args.out / "held-out.json", dict(physics=args.mode, policy_sha256=_sha(policy), seeds=HOLDOUT,
                                           gate=">=3/4 and at least one success in each arm configuration", groups=report))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task", choices=["carry", "push_t"], required=True)
    p.add_argument("--stage", choices=["train", "evaluate", "smoke"], default="train")
    p.add_argument("--mode", choices=["cpu", "musa"], default="musa")
    p.add_argument("--arms", choices=["both", "left", "right"], default="both")
    p.add_argument("--policy-variant", choices=["centroid", "registered", "fiducial"], default="centroid")
    p.add_argument("--tool-profile", choices=["single-pad", "twin-pad"], default="single-pad")
    p.add_argument("--load-profile", choices=["standard", "side-lip"], default="standard")
    p.add_argument("--stance", choices=["original", "bent-elbow"], default="original")
    p.add_argument("--init-policy", type=Path, help="start a new training run with a previously learned policy")
    p.add_argument("--library", type=Path)
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/bimanual-musa/cache"))
    p.add_argument("--device-index", type=int)
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--population", type=int, default=6)
    p.add_argument("--train-seeds", type=int, default=2)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--seed", type=int, default=1731)
    p.add_argument("--wall-seconds", type=float)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if args.load_profile == "side-lip" and args.task != "carry":
        p.error("the side-lip tote is a carry payload")
    if (min(args.image_size, args.action_repeat, args.horizon, args.generations, args.train_seeds, args.batch) < 1
            or args.population < 4 or args.seed < 0 or (args.wall_seconds is not None and args.wall_seconds <= 0)):
        p.error("positive sizes, at least four candidates, and a nonnegative seed are required")
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(args)
    contract = _contract(args)
    checkpoint = args.out / "checkpoint.json"
    if args.stage == "smoke":
        if any(args.out.iterdir()):
            p.error("smoke needs a fresh output directory")
        _save(args.out / "smoke-contract.json", contract)
        runner.episodes("smoke", 701, [_job("prior", PRIOR, args.arms)], keep_trajectory=True)
        return
    if args.stage == "evaluate":
        if not checkpoint.is_file():
            p.error("evaluation requires a trained checkpoint")
        saved = _read(checkpoint)["contract"]
        if _sha(saved) != _sha(contract):
            p.error("evaluation inputs differ from the training contract")
        policy = _read(args.out / "frozen-policy.json")
        if policy["contract_sha256"] != _sha(contract):
            p.error("frozen policy does not match the tracked learning run")
        _evaluate(args, runner, policy)
        return
    if checkpoint.exists():
        if not args.resume:
            p.error("existing training run requires --resume")
        state = _read(checkpoint)
        if _sha(state["contract"]) != _sha(contract):
            p.error("resume contract differs from original sources or task/sampling settings")
        if (args.out / "frozen-policy.json").is_file():
            print("frozen policy already available", flush=True)
            return
    else:
        if args.resume or any(args.out.iterdir()):
            p.error("new training requires a fresh directory")
        initial, initializer = PRIOR.tolist(), None
        if args.init_policy:
            learned = _read(args.init_policy)
            if (learned["task"] != args.task or learned["arms"] != args.arms
                    or learned.get("load_profile", "standard") != args.load_profile
                    or learned.get("tool_profile", "single-pad") != args.tool_profile
                    or learned.get("stance", "original") != args.stance
                    or learned.get("variant", "centroid") != args.policy_variant
                    or list(learned["parameters"]) != list(PARAMETERS) or learned["generations"] < 1):
                p.error("initial policy has a different task, arm condition, actor variant or parameter layout")
            weights = np.asarray(learned["weights"], float)
            if weights.shape != PRIOR.shape or not np.isfinite(weights).all():
                p.error("initial policy weights are invalid")
            initial, initializer = weights.tolist(), dict(file=str(args.init_policy), sha256=_sha(learned))
        state = dict(contract=contract, initializer=initializer, generation=0, mean=initial,
                     std=np.full(len(PRIOR), .55).tolist(), updates=[])
        _save(checkpoint, state)
    _train(args, runner, contract, state)


if __name__ == "__main__":
    main()

"""Run a frozen bimanual policy on new seeds and save the actual wrist-guided trajectories.

python examples/bimanual_policy_replay.py --policy /path/to/trained-run \
    --seed 55001 --out /path/to/new-rollouts --source-dir /path/to/frozen/examples

Both frozen-policy.json and checkpoint.json are required. The source directory
is optional when the installed example files or accompanying source bundle match.
"""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np


MODEL_DEFAULTS = {"tool_profile": "single-pad", "load_profile": "standard", "stance": "original"}


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _load(policy_path, checkpoint_path):
    policy_bytes, checkpoint_bytes = policy_path.read_bytes(), checkpoint_path.read_bytes()
    policy, checkpoint = json.loads(policy_bytes), json.loads(checkpoint_bytes)
    contract = checkpoint["contract"]
    if (policy.get("schema") != "bimanual-visual-policy-v1"
            or policy["contract_sha256"] != _hash(_canonical(contract))
            or policy["task"] != contract["task"] or policy["arms"] != contract["arms"]
            or policy.get("variant", "centroid") != contract.get("policy_variant", "centroid")
            or any(policy[key] != contract.get(key, default)
                   for key, default in MODEL_DEFAULTS.items() if key in policy)
            or policy["parameters"] != contract["parameters"]
            or not 0 < policy["generations"] == checkpoint["generation"] == len(checkpoint["updates"])):
        raise ValueError("frozen policy and training checkpoint do not describe the same completed run")
    scores = np.asarray(policy["selection_returns"], dtype=float)
    if scores.shape != (len(contract["selection_seeds"]), 2) or not np.isfinite(scores).all():
        raise ValueError("incomplete development-seed model selection")
    selected = int(np.argmax(scores.mean(axis=0)))
    expected = (checkpoint["mean"], checkpoint["updates"][-1]["best"])[selected]
    weights = np.asarray(policy["weights"], dtype=float)
    if (policy["selected"] != ("mean", "latest-elite")[selected]
            or weights.shape != (len(contract["parameters"]),) or not np.isfinite(weights).all()
            or not np.array_equal(weights, expected)):
        raise ValueError("frozen weights do not match the winning trained candidate")
    return policy, contract, checkpoint["updates"], _hash(policy_bytes), _hash(checkpoint_bytes)


def _sources(manifest, policy_path, explicit, variant):
    required = {"bimanual_musa_train.py", "_bimanual_visual_policy.py", "bimanual_musa_env.py",
                "bimanual_wrist_render.py"}
    allowed = required | {"_bimanual_image_pose.py", "_bimanual_fiducial.py"}
    if variant == "fiducial":
        required = required | {"_bimanual_fiducial.py"}
    if not required <= set(manifest) <= allowed:
        raise ValueError("invalid example source manifest")
    roots = ([explicit] if explicit else [policy_path.parent / "sources", policy_path.parent / "examples",
                                          policy_path.parent, Path(__file__).resolve().parent])
    for root in roots:
        for directory in (root.resolve(), (root / "examples").resolve()):
            if all((directory / name).is_file() and _hash((directory / name).read_bytes()) == digest
                   for name, digest in manifest.items()):
                for name in allowed:
                    module = sys.modules.get(Path(name).stem)
                    if module is not None and (name not in manifest or Path(module.__file__).resolve() != directory / name):
                        raise ValueError(f"a different version of {name} is already imported")
                sys.path.insert(0, str(directory))
                trainer = importlib.import_module("bimanual_musa_train")
                if Path(trainer.__file__).resolve() != directory / "bimanual_musa_train.py":
                    raise ValueError("cannot bind the original training module")
                return directory, trainer
    raise ValueError("example source hashes differ from the training contract; "
                     "pass --source-dir pointing to this checkpoint's extracted frozen examples")


def _film_files(out, seed, policy, contract, scene):
    stem = Path("episodes") / "replay" / f"{seed}-frozen"
    paths = {suffix: (out / stem).with_suffix("." + suffix) for suffix in ("json", "npz", "xml")}
    receipt = json.loads(paths["json"].read_bytes())
    digests = {suffix: _hash(path.read_bytes()) for suffix, path in paths.items()}
    profiles = {key: contract.get(key, default) for key, default in MODEL_DEFAULTS.items()}
    if (digests["xml"] != scene.xml_sha256 or receipt["scene_sha256"] != digests["xml"]
            or receipt["job"]["weights"] != policy["weights"] or receipt["physics"] != contract["mode"]
            or receipt["seed"] != seed or receipt["task"] != policy["task"]
            or tuple(receipt["arm_dofs"]) != tuple(scene.arm_dofs)
            or any(receipt.get(key, default) != profiles[key] for key, default in MODEL_DEFAULTS.items())
            or receipt.get("fiducials", False) != (contract.get("policy_variant") == "fiducial")
            or receipt["job"]["arms"] != policy["arms"] or receipt["job"]["black"]
            or receipt.get("trajectory_sha256", digests["npz"]) != digests["npz"]):
        raise ValueError(f"episode receipt does not match the frozen policy, seed or files: {stem}")
    with np.load(paths["npz"], allow_pickle=False) as trajectory:
        ticks, times, q = (trajectory[k] for k in ("ticks", "time_s", "qpos"))
        count = int(receipt["decisions"])
        if ("model_sha256" in trajectory and str(trajectory["model_sha256"].item()) != digests["xml"]
                or count < 1 or ticks.shape != (count+1,) or times.shape != ticks.shape
                or q.ndim != 2 or len(q) != len(times) or not np.isfinite(q).all()
                or not np.isfinite(times).all() or not np.all(np.diff(times) > 0)
                or ticks[0] != 0 or not np.all(np.diff(ticks) > 0) or ticks[-1] != receipt["physics_ticks"]
                or trajectory["action"].shape != (count, 14) or not np.isfinite(trajectory["action"]).all()
                or not np.array_equal(trajectory["rewards"], receipt["rewards"])):
            raise ValueError(f"trajectory structure or recorded episode binding is invalid: {stem}")
        if profiles["stance"] == "bent-elbow" and policy["arms"] == "both":
            key = ET.fromstring(scene.xml).find("./keyframe/key[@name='bent-elbow-start']")
            if key is None:
                raise ValueError(f"source model lacks the contracted bent-elbow initial state: {stem}")
            initial = np.asarray(key.attrib["qpos"].split(), dtype=np.float64)
            if initial.shape != q[0].shape or not np.all((q[0] == initial) | (q[0] == initial.astype(np.float32))):
                raise ValueError(f"first recorded pose differs from the source model keyframe: {stem}")
    return dict(seed=seed, **profiles, success=receipt["success"], return_sum=receipt["return_sum"], decisions=count,
                physical_ticks=receipt["physics_ticks"], simulated_seconds=float(times[-1]),
                sensor_rgb_sha256=receipt["sensor_rgb_sha256"], physics=receipt["physics"], arm_dofs=receipt["arm_dofs"],
                files={k: str(p.relative_to(out)) for k, p in paths.items()}, file_sha256=digests)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", type=Path, required=True, help="training directory or frozen-policy.json")
    parser.add_argument("--checkpoint", type=Path, help="defaults to checkpoint.json beside the frozen policy")
    parser.add_argument("--source-dir", type=Path, help="directory containing the digest-matched example sources")
    parser.add_argument("--seed", "--seeds", type=int, nargs="+", default=[55001], help="one or more environment seeds")
    parser.add_argument("--out", type=Path, required=True, help="fresh directory for episode files and replay.json")
    parser.add_argument("--library", type=Path, help="native Vulkan renderer library")
    parser.add_argument("--device-index", type=int, help="Vulkan adapter index")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/bimanual-musa/cache"))
    args = parser.parse_args(argv)
    if min(args.seed) < 0 or len(set(args.seed)) != len(args.seed):
        parser.error("choose distinct nonnegative seeds")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error("output must be a new or empty directory")
    path = (args.policy / "frozen-policy.json" if args.policy.is_dir() else args.policy).resolve()
    cp = (args.checkpoint or path.with_name("checkpoint.json")).resolve()
    policy, contract, updates, p_hash, cp_hash = _load(path, cp)
    variant = contract.get("policy_variant", "centroid")
    directory, trainer = _sources(contract["sources"], path, args.source_dir, variant)
    environment = importlib.import_module("bimanual_musa_env")
    if Path(environment.__file__).resolve() != directory / "bimanual_musa_env.py":
        raise ValueError("cannot bind the source model generator to the training contract")
    model_options = {key: contract[key] for key in MODEL_DEFAULTS if key in contract}
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    run = SimpleNamespace(**dict(contract, out=args.out, policy_variant=variant,
                                tool_profile=contract.get("tool_profile", "single-pad"),
                                load_profile=contract.get("load_profile", "standard"),
                                stance=contract.get("stance", "original"),
                                cache_dir=args.cache_dir, batch=1, wall_seconds=None,
                                library=args.library, device_index=args.device_index))
    runner = trainer.Runner(run)
    seeds_used = {s for update in updates for s in update["seeds"]}
    report = dict(schema="bimanual-policy-replay-v1", frozen_policy_sha256=p_hash, checkpoint_sha256=cp_hash,
                  contract_sha256=policy["contract_sha256"], example_sources=str(directory), source_sha256=contract["sources"],
                  task=run.task, arms=policy["arms"], physics=run.mode, variant=variant,
                  tool_profile=run.tool_profile, load_profile=run.load_profile, stance=run.stance,
                  image_size=run.image_size, action_repeat=run.action_repeat, horizon=run.horizon, episodes=[])
    target = args.out / "replay.json"
    for seed in args.seed:
        job = trainer._job("frozen", policy["weights"], policy["arms"])
        runner.episodes("replay", seed, [job], keep_trajectory=True)
        expected_scene = environment.make_scene(run.task, seed, trainer._dofs(seed), **model_options)
        episode = _film_files(args.out, seed, policy, contract, expected_scene)
        episode["seed_partition"] = ("training" if seed in seeds_used else "selection" if seed in contract["selection_seeds"]
                                     else "original_holdout" if seed in contract["holdout_seeds"] else "new")
        report["episodes"].append(episode)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(_canonical(report))
        tmp.replace(target)
        print(f"seed {seed}: success={episode['success']}, files={args.out / episode['files']['json']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

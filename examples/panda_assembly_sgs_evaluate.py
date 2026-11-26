"""Evaluate a frozen dual-Panda assembly policy on a reproducible reset bank."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from _panda_assembly_scene import build_scene
from panda_assembly_sgs_train import Geometry, reset_bank, run_batch, save


STEPS = 525
DEFAULT_SEED = 290317


def evaluate(policy_path, mode, out, *, urdf=None, cache=None, seed=DEFAULT_SEED, count=6):
    if mode not in ("cpu", "musa") or not 1 <= count <= 6 or seed < 0:
        raise ValueError("expected cpu/musa, 1–6 resets and a nonnegative bank seed")
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"evaluation needs a new output directory: {out}")
    blob = Path(policy_path).read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    policy = json.loads(blob)
    task = policy["task"]
    if task not in ("pin", "screw_proxy"):
        raise ValueError(f"unsupported policy task: {task}")
    optimizer = policy["optimizer"]
    mean = np.asarray(optimizer["mean"], dtype=float)
    if mean.shape != (3,) or not np.isfinite(mean).all():
        raise ValueError("the frozen mean must contain three finite controller parameters")
    if optimizer["generation"] < 1 or optimizer["pending"] is not None:
        raise ValueError("expected a policy after a completed optimizer update")
    scene = build_scene(task, urdf_path=urdf)
    if scene.task != task or scene.xml_sha256 != policy["scene_sha256"]:
        raise ValueError("evaluation model does not match the checkpoint task/scene digest")
    comp = scene.compile()
    if (comp.mj_model.nq, comp.mj_model.nv, comp.mj_model.nu) != (21, 20, 14):
        raise ValueError("expected two actuated Panda arms and one free assembly part")

    starts = reset_bank(task, seed, count)
    rows, hardware = run_batch(task, scene, comp, Geometry(scene, comp), starts, [mean]*count,
                               mode, cache, STEPS, monitor="trained")
    out.mkdir(parents=True, exist_ok=False)
    (out/"scene.xml").write_bytes(scene.xml.encode())
    episodes = []
    for start, row in zip(starts, rows, strict=True):
        hist = row.pop("history")
        times = np.array([t for t, _ in hist], dtype=float)
        poses = np.stack([q for _, q in hist])
        if (poses.shape != (STEPS//3+1, comp.mj_model.nq) or times.shape != (len(poses),)
                or not np.isfinite(poses).all() or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0.) or not np.isclose(times[-1]-times[0], STEPS*scene.timestep)):
            raise RuntimeError(f"invalid recorded trajectory for reset {start['id']}")
        stem = f"episode-{start['id']:02d}"
        trajectory = out/(stem+".npz")
        np.savez_compressed(trajectory, time_s=times, qpos=poses, model_sha256=scene.xml_sha256,
                            policy_sha256=digest, seed=seed, reset_id=start["id"])
        receipt = dict(**row, task=task, policy_sha256=digest, scene_sha256=scene.xml_sha256,
                       trajectory=trajectory.name, trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest())
        save(out/(stem+".json"), receipt)
        episodes.append(dict(id=start["id"], success=row["success"], receipt=stem+".json",
                             trajectory=trajectory.name))
    result = dict(task=task, mode=mode, seed=seed, count=count, successes=sum(r["success"] for r in rows),
                  steps=STEPS, timestep=scene.timestep, record_every_steps=3, device_batch=hardware,
                  policy_file=Path(policy_path).name, policy_sha256=digest, frozen_mean=mean.tolist(),
                  policy_generation=optimizer["generation"], scene_sha256=scene.xml_sha256,
                  urdf_sha256=scene.urdf_sha256, episodes=episodes)
    save(out/"result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--mode", choices=("musa", "cpu"), default="musa")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--urdf", type=Path, help="Panda URDF used to build the checkpoint scene")
    parser.add_argument("--cache", type=Path, help="optional backend cache directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="evaluation reset-bank seed")
    parser.add_argument("--count", type=int, choices=range(1, 7), default=6)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    report = evaluate(args.policy, args.mode, args.out, urdf=args.urdf, cache=args.cache,
                      seed=args.seed, count=args.count)
    print(f"{report['task']} {args.mode}: {report['successes']}/{args.count} successes; {args.out}")

"""Reward-based pickup policy search and blind evaluation for dual Panda assembly."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from _panda_assembly_sgs import StateSampler
from _panda_grasp_insert_scene import build_scene
from _panda_pickup_policy import PickupPolicy
from _panda_pickup_resets import reset_bank
from _panda_pickup_rollout import run_batch


def write_json(path, obj):
    def scalar(value):
        if isinstance(value, (np.ndarray, np.generic)):
            return value.tolist()
        raise TypeError(type(value).__name__)

    with path.open("x", encoding="utf8") as out:
        json.dump(obj, out, indent=2, allow_nan=False, default=scalar)
        out.write("\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def early_population():
    # Latin-hypercube exploration plus the predecessor's uncorrected command.
    rng = np.random.default_rng(6121)
    samples = np.column_stack([(rng.permutation(3) + rng.random(3))/3 for _ in range(2)])
    return [(0., 0.), *map(tuple, .20 + 1.1*samples)]


def refined_population(center):
    rng = np.random.default_rng(8703)
    draws = rng.normal(size=(3, 2))
    return [center, *map(tuple, np.clip(np.asarray(center) + .24*draws, 0., 1.35))]


def rank(rows):
    """Lexicographic return over complete tasks, phase gates, and release accuracy."""
    success = sum(all(r["score"]["gates"].values()) for r in rows)
    gates = sum(sum(r["score"]["gates"].values()) for r in rows)
    errors = [r["score"]["phases"]["clear"]["lateral_m"][1] for r in rows]
    return (success, gates, -float(np.mean(np.minimum(errors, .05))))


def progress(tick, total):
    if tick % 1500 == 0 or tick == total:
        print(f"  physics {tick}/{total}", flush=True)


def generation(scene, compiled, bank, policies, indices, mode, cache):
    cases = [(bank[i], PickupPolicy(gain)) for gain in policies for i in indices]
    results, audit = run_batch(scene, compiled, cases, mode=mode, cache=cache, progress=progress)
    candidates = []
    n = len(indices)
    for j, gain in enumerate(policies):
        rows = results[j*n:(j+1)*n]
        candidates.append(dict(parameters=list(gain), rank=rank(rows), cases=[
            dict(id=bank[i]["id"], **{key: r[key] for key in ("score", "diagnostics")})
            for i, r in zip(indices, rows, strict=True)]))
    winner = max(range(len(policies)), key=lambda j: candidates[j]["rank"])
    print(mode, "training ranks", [x["rank"] for x in candidates], "selected", winner, flush=True)
    return dict(indices=indices, candidates=candidates, winner=winner, audit=audit)


def protocol(scene):
    return dict(schema="panda-random-pickup-experiment-v1", scene_sha256=scene.xml_sha256,
                train=reset_bank("train"), holdout=reset_bank("holdout"),
                initial_population=early_population(), refine_seed=8703, refine_std=.24,
                strategy="CPU train-bank search; 4 S4000 candidates on 2 distinct SGS-sampled train starts",
                ranking=["complete task count", "passed phase gate count", "negative mean clipped clear lateral error"],
                validation=["fixed nominal", "frozen trained checkpoint", "unit observation gains"],
                recording="first holdout with matched open-jaw control regardless of success")


def train(args, scene, compiled):
    args.out.mkdir(parents=True, exist_ok=False)
    plan = protocol(scene)
    write_json(args.out/"protocol.json", plan)
    (args.out/"scene.xml").write_bytes(scene.xml.encode())
    bank = plan["train"]
    initial = generation(scene, compiled, bank, plan["initial_population"],
                         list(range(len(bank))), "cpu", args.cache)
    write_json(args.out/"cpu-training.json", initial)
    sampler = StateSampler(len(bank), seed=803, mode="sgs")
    for candidate in initial["candidates"]:
        sampler.update(initial["indices"], [all(x["score"]["gates"].values()) for x in candidate["cases"]])
    indices = []
    while len(indices) < 2:
        picked = int(sampler.sample(1)[0])
        if picked not in indices:
            indices.append(picked)
    center = initial["candidates"][initial["winner"]]["parameters"]
    policies = refined_population(center)
    write_json(args.out/"refinement-input.json", dict(indices=indices, probabilities=sampler.probabilities(),
                                                      parameters=policies, center=center))
    final = generation(scene, compiled, bank, policies, indices, "musa", args.cache)
    write_json(args.out/"musa-training.json", final)
    checkpoint = args.out/"policy.json"
    PickupPolicy(policies[final["winner"]]).save(checkpoint)
    write_json(args.out/"frozen.json", dict(scene_sha256=scene.xml_sha256, policy_sha256=sha(checkpoint),
                                           protocol_sha256=sha(args.out/"protocol.json"),
                                           training_sha256=sha(args.out/"musa-training.json")))
    print("frozen", sha(checkpoint), flush=True)


def validate(args, scene, compiled):
    train_dir = args.training
    checkpoint = train_dir/"policy.json"
    frozen = json.loads((train_dir/"frozen.json").read_text())
    if (frozen["scene_sha256"] != scene.xml_sha256 or frozen["policy_sha256"] != sha(checkpoint)
            or frozen["protocol_sha256"] != sha(train_dir/"protocol.json")
            or frozen["training_sha256"] != sha(train_dir/"musa-training.json")):
        raise ValueError("frozen run does not match this evaluation")
    policy = PickupPolicy.load(checkpoint)
    args.out.mkdir(parents=True, exist_ok=False)
    bank = json.loads((train_dir/"protocol.json").read_text())["holdout"]
    variants = dict(nominal=PickupPolicy(), trained=policy, unit=PickupPolicy((1., 1.)))
    cases = [(case, p) for case in bank for p in variants.values()]
    closed = 1
    opened = len(cases)
    cases.append((bank[0], policy))
    rows, audit = run_batch(scene, compiled, cases, mode="musa", cache=args.cache, record=(closed, opened),
                            open_reference={opened: closed}, progress=progress)
    saved = {}
    for i, name in ((closed, "closed"), (opened, "open")):
        path = args.out/(name+".npz")
        np.savez_compressed(path, **rows[i]["trace"])
        saved[name] = sha(path)
    evaluations = {}
    for j, (name, p) in enumerate(variants.items()):
        subset = rows[j:opened:len(variants)]
        evaluations[name] = dict(parameters=p.parameters, successes=rank(subset)[0], total=len(bank), cases=[
            dict(id=case["id"], success=all(row["score"]["gates"].values()),
                 score=row["score"], diagnostics=row["diagnostics"])
            for case, row in zip(bank, subset, strict=True)])
    write_json(args.out/"summary.json", dict(frozen=frozen, evaluations=evaluations, audit=audit,
                                            traces=saved, recorded_case=bank[0]["id"],
                                            open_control=rows[opened]["position_control"]))
    print({k: v["successes"] for k, v in evaluations.items()}, "out of", len(bank), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "validate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training", type=Path, help="frozen training directory for validation")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--urdf", type=Path)
    args = parser.parse_args()
    if args.stage == "validate" and args.training is None:
        parser.error("validate requires --training")
    scene = build_scene(urdf_path=args.urdf)
    (train if args.stage == "train" else validate)(args, scene, scene.compile())

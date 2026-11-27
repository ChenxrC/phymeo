"""Train a residual pickup policy and compare paired deterministic rollouts on reserved domains."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from _panda_pickup_rl_actor import ResidualPickupActorCritic
from _panda_pickup_rl_domains import make_batch
from _panda_pickup_rl_optimize import update_policy


def save_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def run(args):
    if min(args.worlds, args.updates, args.decisions) < 1:
        raise ValueError("positive work dimensions required")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    if args.backend == "musa":
        import torch_musa  # registers torch.musa
        from _panda_pickup_rl_musa import run_musa
        torch.musa.manual_seed_all(args.seed)
        def rollout(domain, actor, mode):
            return run_musa(domain, actor, decisions=args.decisions, mode=mode, cache=args.cache)
    else:
        from _panda_pickup_rl_native import run_native
        torch.set_num_threads(1)
        def rollout(domain, actor, mode):
            return run_native(domain, actor, decisions=args.decisions, mode=mode)
    xml = args.scene.read_text(encoding="utf-8")
    student = args.handoffs / "student.npz"
    device = "cpu" if args.backend == "native" else "musa"
    policy = ResidualPickupActorCritic.from_student(student, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    result = dict(schema="panda-residual-pickup-ppo-v1", backend=args.backend,
                  seed=args.seed, worlds=args.worlds, decisions=args.decisions, repeat=8,
                  updates=[], evaluation={}, status="training", optimizer=dict(lr=3e-4, epochs=3, minibatch=256),
                  prior_sha256=hashlib.sha256(student.read_bytes()).hexdigest())
    start = time.monotonic()
    for i in range(args.updates):
        domain = make_batch(xml, args.handoffs, "train", i, worlds=args.worlds, batch=i)
        name = f"train-{i:02d}"
        save_json(args.out / (name + "-domain.json"), domain.manifest)
        records, _, stats = rollout(domain, policy, "train")
        metrics = update_policy(torch, policy, optimizer, records)
        result["updates"].append(dict(group=i, rollout=stats, optimization=metrics))
        save_json(args.out / "result.json", result)
        print(name, "return", stats["mean_return"], "actor change", metrics["actor_max_parameter_change"], flush=True)
    checkpoint = args.out / "candidate.pt"
    torch.save(policy.checkpoint(), checkpoint)
    result["candidate_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # Evaluate the serialized policy, with deterministic actions and identical starts in both arms.
    policy = ResidualPickupActorCritic.from_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=True),
                                                       student, device).eval()
    for split in ("select", "blind"):
        pairs = []
        for group in (0, 1):
            domain = make_batch(xml, args.handoffs, split, group, worlds=args.worlds)
            name = f"{split}-{group}"
            save_json(args.out / (name + "-domain.json"), domain.manifest)
            pair = {}
            for mode in ("prior", "candidate"):
                _, scores, stats = rollout(domain, policy, mode)
                if len(scores) != args.worlds:
                    raise ValueError("one independent physical score per world required")
                pair[mode] = [bool(row["success"]) for row in scores]
                save_json(args.out / (name + "-" + mode + ".json"), dict(rollout=stats, scores=scores))
            prior, candidate = pair["prior"], pair["candidate"]
            pairs.append(dict(group=group, prior=sum(prior), candidate=sum(candidate),
                              gained=sum(b and not a for a, b in zip(prior, candidate)),
                              lost=sum(a and not b for a, b in zip(prior, candidate))))
            print(name, pairs[-1], flush=True)
        result["evaluation"][split] = pairs
        result[split + "_improved"] = (all(p["candidate"] >= p["prior"] for p in pairs)
                                      and sum(p["gained"] - p["lost"] for p in pairs) > 0)
        save_json(args.out / "result.json", result)
    result.update(status="finished", elapsed_seconds=time.monotonic()-start,
                  cleared_paired_screen=result["select_improved"] and result["blind_improved"])
    save_json(args.out / "result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("examples/checkpoints/panda-reach-factors-s4000/scene.xml"))
    parser.add_argument("--handoffs", type=Path, default=Path("examples/checkpoints/panda-pickup-from-reach-s4000"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("native", "musa"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--decisions", type=int, default=160)
    parser.add_argument("--seed", type=int, default=271144)
    run(parser.parse_args())

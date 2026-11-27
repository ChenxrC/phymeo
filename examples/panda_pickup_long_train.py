"""Resumable finite-horizon residual pickup PPO on native CPU or resident S4000 physics."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import mujoco as mj
import numpy as np
import torch

from _panda_pickup_rl_actor import ResidualPickupActorCritic
from _panda_pickup_rl_domains import make_batch
from _panda_pickup_rl_optimize import update_policy
from panda_pickup_from_reach import DT, sha


HERE = Path(__file__).resolve().parent
TRAIN_BATCH_START, SELECT_BATCH, BLIND_BATCH = 1000, 510, 610
SOURCE_FILES = ("panda_pickup_long_train.py", "_panda_pickup_rl_optimize.py", "_panda_pickup_rl_actor.py",
                "_panda_pickup_rl_reward.py", "_panda_pickup_rl_domains.py", "_panda_pickup_rl_evaluate.py",
                "_panda_pickup_loads.py", "_panda_pickup_student.py", "panda_pickup_from_reach.py",
                "_panda_grasp_insert_metrics.py", "_panda_grasp_insert_control.py", "panda_state_ppo.py")


def _utc(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def _json_bytes(data):
    return json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _atomic(path, write):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    if os.name == "posix":
        handle = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def _json(path, data):
    if path.exists():
        if _json_bytes(json.loads(path.read_text(encoding="utf8"))) != _json_bytes(data):
            raise ValueError(f"immutable result differs at {path}")
    else:
        _atomic(path, lambda f: f.write(_json_bytes(data) + b"\n"))


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


@contextmanager
def _writer(directory):
    """The OS releases this lock on process death. A stale PID file cannot strand a run."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".trainer.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt
                if lock.tell() == 0:
                    lock.write(b"0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"another trainer holds the writer lock at {directory}") from exc
        yield


def _rng(backend):
    generator, keys, pos, gaussian, cached = np.random.get_state()
    return dict(python=random.getstate(), numpy=(generator, keys.tolist(), pos, gaussian, cached),
                torch_cpu=torch.get_rng_state(), torch_musa=torch.musa.get_rng_state_all() if backend == "musa" else [])


def _restore_rng(state, backend):
    random.setstate(state["python"])
    generator, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((generator, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state["torch_cpu"])
    if backend == "musa":
        torch.musa.set_rng_state_all(state["torch_musa"])


def _actor_hash(payload):
    h = hashlib.sha256(_json_bytes({k: v for k, v in payload.items() if k != "state"}))
    for name, tensor in sorted(payload["state"].items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(_json_bytes([name, a.dtype.str, a.shape]))
        h.update(a.tobytes())
    return h.hexdigest()


def _configuration(args):
    fields = ("backend", "seed", "worlds", "decisions", "repeat", "updates", "eval_every", "eval_worlds",
              "epochs", "minibatch", "lr", "arm_std", "finger_std", "rho", "threads")
    if min(getattr(args, k) for k in ("updates", "worlds", "decisions", "repeat", "eval_every", "eval_worlds",
                                      "epochs", "minibatch", "threads")) < 1 or args.decisions * args.repeat < 2:
        raise ValueError("positive training dimensions and at least two solved inputs required")
    if not (0 <= args.seed < 2**32 and 0 <= args.rho < 1 and math.isfinite(args.lr) and args.lr > 0):
        raise ValueError("invalid training seed, learning rate or arm correlation")
    if not all(s is None or (math.isfinite(s) and s > 0) for s in (args.arm_std, args.finger_std)):
        raise ValueError("exploration deviations must be finite and positive")
    files = [HERE / name for name in (*SOURCE_FILES, f"_panda_pickup_rl_{args.backend}.py")]
    if args.backend == "musa":
        files += [HERE / "_panda_ppo_reach.py", HERE / "panda_pickup_musa_compare.py",
                  HERE.parent / "src/phymeo/backends/mujoco_musa_resident.py"]
    return dict(schema="pickup-long-ppo-v1", **{k: getattr(args, k) for k in fields},
                scene_sha256=sha(args.scene), prior_sha256=sha(args.handoffs / "student.npz"),
                banks={n: sha(args.handoffs / f"{n}.npz") for n in ("factor-update", "validation")},
                source_sha256={f.name: hashlib.sha256(f.read_text(encoding="utf8").encode()).hexdigest() for f in files},
                versions=dict(torch=str(torch.__version__), numpy=np.__version__, mujoco=mj.__version__),
                train_batch_start=TRAIN_BATCH_START, select_batch=SELECT_BATCH, blind_batch=BLIND_BATCH,
                evaluation_groups=[0, 1], selector="maximize successful worlds, then total loaded dwell seconds, then earliest update")


def _rollout(args, domain, policy, mode):
    if args.backend == "musa":
        from _panda_pickup_rl_musa import run_musa
        return run_musa(domain, policy, decisions=args.decisions, repeat=args.repeat, mode=mode, cache=args.cache)
    from _panda_pickup_rl_native import run_native
    return run_native(domain, policy, decisions=args.decisions, repeat=args.repeat, mode=mode)


def _evaluation(args, xml, policy, split, number, actor_sha, run_sha):
    """One immutable artifact per split/group/policy. Replay a committed receipt on restart."""
    outcomes = []
    for group in (0, 1):
        batch = SELECT_BATCH if split == "select" else BLIND_BATCH
        domain = make_batch(xml, args.handoffs, split, group, args.eval_worlds, batch=batch)
        common = dict(run_sha256=run_sha, split=split, group=group, batch=batch,
                      resets_sha256=domain.manifest["resets_sha256"], model_sha256=domain.manifest["xml_sha256"])
        pair = {}
        for mode in ("prior", "candidate"):
            tag = "prior" if mode == "prior" else f"u{number:05d}"
            path = args.out / f"{split}-g{group}-{tag}.json"
            identity = dict(**common, policy=mode, policy_sha256=args.prior_sha if mode == "prior" else actor_sha)
            if path.exists():
                item = json.loads(path.read_text(encoding="utf8"))
                if item["identity"] != identity:
                    raise ValueError(f"evaluation binding differs at {path}")
            else:
                _, scores, stats = _rollout(args, domain, policy, mode)
                if len(scores) != args.eval_worlds:
                    raise ValueError("expected one independent score per physical world")
                item = dict(identity=identity, rollout=stats, scores=scores, domain=domain.manifest)
                _json(path, item)
            pair[mode] = item["scores"]
        p, c = pair["prior"], pair["candidate"]
        outcomes.append(dict(group=group, worlds=len(c), prior=sum(s["success"] for s in p),
                             candidate=sum(s["success"] for s in c),
                             gained=sum(b["success"] and not a["success"] for a, b in zip(p, c, strict=True)),
                             lost=sum(a["success"] and not b["success"] for a, b in zip(p, c, strict=True)),
                             loaded_dwell_s=sum(s["longest_loaded_dwell_s"] for s in c)))
    return outcomes


def _rank(item):
    return (sum(g["candidate"] for g in item["groups"]), sum(g["loaded_dwell_s"] for g in item["groups"]), -item["update"])


def _paired_improvement(groups):
    return (all(g["candidate"] >= g["prior"] for g in groups)
            and sum(g["candidate"] - g["prior"] for g in groups) > 0)


def _snapshot(path, payload):
    if path.exists():
        if _actor_hash(_load(path)) != _actor_hash(payload):
            raise ValueError(f"immutable snapshot differs at {path}")
    else:
        _atomic(path, lambda f: torch.save(payload, f))


def run(args):
    with _writer(args.out):
        config = _configuration(args)
        _json(args.out / "config.json", config)
        binding = hashlib.sha256(_json_bytes(config)).hexdigest()
        args.prior_sha = config["prior_sha256"]
        student = args.handoffs / "student.npz"
        xml = args.scene.read_text(encoding="utf8")
        latest = args.out / "latest.pt"
        device = "cpu" if args.backend == "native" else "musa:0"
        if args.backend == "musa":
            import torch_musa  # noqa: F401 -- register the device
            torch.musa.set_device(0)
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        if args.backend == "musa":
            torch.musa.manual_seed_all(args.seed)
        policy = ResidualPickupActorCritic.from_student(student, device, arm_std=args.arm_std,
                                                      finger_std=args.finger_std, arm_rho=args.rho)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
        if latest.exists():
            state = _load(latest)
            if state["run_sha256"] != binding or state["config"] != config:
                raise ValueError("resume refused: checkpoint training contract differs")
            policy.load_state_dict(state["actor"]["state"])
            optimizer.load_state_dict(state["optimizer"])
        else:
            state = dict(config=config, run_sha256=binding, actor=policy.checkpoint(), optimizer=optimizer.state_dict(),
                         next_update=1, completed_world_episodes=0, completed_train_world_steps=0,
                         next_train_batch=TRAIN_BATCH_START, rng=_rng(args.backend), start_timestamp=time.time(), last_metric=None)
            _atomic(latest, lambda f: torch.save(state, f))

        # latest.pt commits the update before any per-update outputs. Replay these inexpensive
        # projections and complete any pending deterministic evaluation before starting another update.
        while True:
            number = state["next_update"] - 1
            delta_steps = args.worlds * args.decisions * args.repeat
            if (state["completed_train_world_steps"] != number * delta_steps or state["next_train_batch"] != TRAIN_BATCH_START + number
                    or state["completed_world_episodes"] != number * args.worlds or not 0 <= number <= args.updates):
                raise ValueError("inconsistent checkpoint episode/step counters")
            if number:
                _json(args.out / f"update-{number:05d}.json", state["last_metric"])
                if number % args.eval_every == 0 or number == args.updates:
                    snapshot = args.out / f"actor-{number:05d}.pt"
                    _snapshot(snapshot, state["actor"])
                    actor_sha = _actor_hash(state["actor"])
                    groups = _evaluation(args, xml, policy.eval(), "select", number, actor_sha, binding)
                    _json(args.out / f"selection-{number:05d}.json", dict(update=number, groups=groups, actor_sha256=actor_sha))
            if number == args.updates:
                break
            _restore_rng(state["rng"], args.backend)
            index = number + 1
            # Revisit nominal dynamics every third update; fresh poses/velocities in every batch.
            group = 0 if number % 3 == 0 else number
            domain = make_batch(xml, args.handoffs, "train", group, args.worlds, batch=state["next_train_batch"])
            records, _, stats = _rollout(args, domain, policy.train(), "train")
            metrics = update_policy(torch, policy, optimizer, records, epochs=args.epochs, minibatch=args.minibatch)
            if metrics["samples"] != args.worlds * args.decisions or stats["physics_world_steps"] != delta_steps:
                raise RuntimeError("training update has incomplete physical or policy samples")
            metric = dict(update=index, rollout=stats, optimization=metrics, domain=domain.manifest,
                          training_world_steps=index * delta_steps, training_world_seconds=index * delta_steps * DT,
                          completed_utc=_utc(time.time()))
            state.update(actor=policy.checkpoint(), optimizer=optimizer.state_dict(), next_update=index+1,
                         next_train_batch=TRAIN_BATCH_START+index, completed_world_episodes=index * args.worlds,
                         completed_train_world_steps=index * delta_steps, last_metric=metric, rng=_rng(args.backend))
            _atomic(latest, lambda f: torch.save(state, f))
            print(f"Update {index}/{args.updates}, {index*delta_steps} training world steps, mean return {stats['mean_return']:.5g}",
                  flush=True)

        selection = [json.loads((args.out / f"selection-{n:05d}.json").read_text(encoding="utf8"))
                     for n in range(1, args.updates + 1) if n % args.eval_every == 0 or n == args.updates]
        best = max(selection, key=_rank)
        candidate = args.out / f"actor-{best['update']:05d}.pt"
        payload = _load(candidate)
        if _actor_hash(payload) != best["actor_sha256"]:
            raise ValueError("selected snapshot content differs from evaluated policy")
        policy.load_state_dict(payload["state"])
        blind = _evaluation(args, xml, policy.eval(), "blind", best["update"], best["actor_sha256"], binding)
        final_path = args.out / "final.json"
        if final_path.exists():
            final = json.loads(final_path.read_text(encoding="utf8"))
            if (final["run_sha256"] != binding or final["select"] != best or final["blind"] != blind
                    or final["selected_file_sha256"] != sha(candidate)):
                raise ValueError("final result differs from frozen evaluated evidence")
        else:
            end = time.time()
            evaluation_steps = (len(selection) + 3) * 2 * args.eval_worlds * args.decisions * args.repeat
            total = state["completed_train_world_steps"] + evaluation_steps
            final = dict(status="finished", backend=args.backend, run_sha256=binding, selected_file=candidate.name,
                         selected_file_sha256=sha(candidate), selector=config["selector"], select=best, blind=blind,
                         blind_scope="Fresh fixed perturbations on previously measured validation handoffs.",
                         training_world_steps=state["completed_train_world_steps"], training_world_episodes=state["completed_world_episodes"],
                         evaluation_world_steps=evaluation_steps, total_world_steps=total, total_simulated_world_seconds=total*DT,
                         started_utc=_utc(state["start_timestamp"]), finished_utc=_utc(end), elapsed_wall_seconds=end-state["start_timestamp"],
                         blind_improved=sum(g["candidate"] for g in blind) > sum(g["prior"] for g in blind),
                         passed_paired_screen=_paired_improvement(best["groups"]) and _paired_improvement(blind))
            _json(final_path, final)
        print("FINAL", json.dumps(final, sort_keys=True), flush=True)
        return final


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("native", "musa"), required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--scene", type=Path, default=HERE / "checkpoints/panda-reach-factors-s4000/scene.xml")
    p.add_argument("--handoffs", type=Path, default=HERE / "checkpoints/panda-pickup-from-reach-s4000")
    p.add_argument("--cache", type=Path)
    p.add_argument("--arm-std", type=float)
    p.add_argument("--finger-std", type=float)
    p.add_argument("--rho", type=float, default=0.)
    p.add_argument("--lr", type=float, default=3e-4)
    for name, value in (("seed", 271145), ("updates", 120), ("worlds", 32), ("decisions", 160), ("repeat", 8),
                        ("eval-every", 12), ("eval-worlds", 32), ("epochs", 3), ("minibatch", 256), ("threads", 1)):
        p.add_argument("--" + name, type=int, default=value)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(arguments())

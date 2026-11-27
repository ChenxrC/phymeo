"""Paired native evaluation of initialization-time pickup exploration on training starts."""
import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from _panda_pickup_loads import LIFT_M
from _panda_pickup_rl_actor import PRIOR_SHA256, ResidualPickupActorCritic
from _panda_pickup_rl_domains import make_batch
from _panda_pickup_rl_native import run_native
from panda_pickup_ppo import save_json


WORLDS, DECISIONS, BATCH, SEED = 16, 160, 401, 271144
CHECKPOINTS = Path(__file__).resolve().parent / "checkpoints"
BRANCHES = ("prior", "original_noise", "attenuated_fingers")


class ExplorationReplay(torch.nn.Module):
    """Inject predeclared common random numbers while preserving state feedback on each branch."""
    def __init__(self, policy, branch, draws):
        super().__init__()
        self.policy, self.branch, self.draws = policy, branch, draws
        self.finger_delta = []

    def observation(self, features, velocity):
        return self.policy.observation(features, velocity)

    def sample(self, obs, deterministic=False):
        if not deterministic:
            raise ValueError("Use the native evaluation path for this fixed-draw diagnostic")
        step = len(self.finger_delta)
        if self.branch == "prior":
            action = self.policy.prior_action(obs[:, :21])
            delta = torch.zeros_like(action[:, 14:])
            raw = logp = value = None
        else:
            prior, mean, log_std, value = self.policy._distribution(obs)
            if self.branch == "attenuated_fingers":
                log_std = log_std.clone()
                log_std[14:] += math.log(.2)
            raw = mean + self.draws[step] * log_std.exp()
            action, logp = self.policy._transport(prior, raw, mean, log_std)
            # Compare against the unperturbed command evaluated at this exact state.
            delta = action[:, 14:] - self.policy.prior_action(obs[:, :21])[:, 14:]
        self.finger_delta.append(delta.detach())
        return action, raw, logp, value


def gates(score):
    return dict(bilateral_handle=score["first_bilateral_handle_s"] is not None,
                lift=score["peak_lift_m"] >= LIFT_M, loaded_pickup=score["success"])


def paired_counts(before, after):
    result = {}
    for key in before[0]:
        pairs = [(a[key], b[key]) for a, b in zip(before, after, strict=True)]
        result[key] = dict(before=sum(a for a, _ in pairs), after=sum(b for _, b in pairs),
                           gained=sum(not a and b for a, b in pairs), lost=sum(a and not b for a, b in pairs))
    return result


def run(out):
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    directory = CHECKPOINTS / "panda-pickup-from-reach-s4000"
    xml = (CHECKPOINTS / "panda-reach-factors-s4000/scene.xml").read_text(encoding="utf-8")
    actor = ResidualPickupActorCritic.from_student(directory / "student.npz", "cpu").eval()
    draws = torch.randn((DECISIONS, WORLDS, 16), generator=torch.Generator().manual_seed(SEED))
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema="panda-pickup-training-exploration-v1", backend="native MuJoCo CPU Euler",
                  split="train", batch=BATCH, seed=SEED, decisions=DECISIONS, repeat=8, worlds_per_group=WORLDS,
                  prior_sha256=PRIOR_SHA256, trained_updates=0, raw_std=actor.log_std.detach().exp().tolist(),
                  common_standard_normal_sha256=hashlib.sha256(draws.numpy().tobytes()).hexdigest(),
                  finger_noise_factor=dict(original_noise=1., attenuated_fingers=.2),
                  action_delta="Per finger, realized action minus published prior on its own current state. "
                               "Absolute jaw target conversion: 17.5 mm per normalized unit.",
                  groups=[])
    for group in (0, 1):
        domain = make_batch(xml, directory, "train", group, worlds=WORLDS, batch=BATCH)
        manifest = f"train-{group}-domain.json"
        save_json(out / manifest, domain.manifest)
        entry = dict(group=group, resets_sha256=domain.manifest["resets_sha256"], domain_file=manifest, branches={})
        all_gates = {}
        for branch in BRANCHES:
            wrapper = ExplorationReplay(actor, branch, draws)
            _, scores, stats = run_native(domain, wrapper, decisions=DECISIONS, mode="candidate")
            if len(scores) != WORLDS or len(wrapper.finger_delta) != DECISIONS:
                raise RuntimeError("Incomplete exploration diagnostic rollout")
            delta = torch.stack(wrapper.finger_delta).numpy()
            absolute = np.abs(delta)
            all_gates[branch] = [gates(score) for score in scores]
            worlds = []
            for w, score in enumerate(scores):
                row = domain.manifest["rows"][w]
                worlds.append(dict(world=w, source_world=row["source_world"], reset_sha256=row["reset_sha256"],
                                   score=score, mean_abs_normalized_finger_delta=absolute[:, w].mean(axis=0).tolist(),
                                   mean_abs_finger_target_delta_mm=(17.5*absolute[:, w].mean(axis=0)).tolist()))
            entry["branches"][branch] = dict(
                counts={key: sum(g[key] for g in all_gates[branch]) for key in all_gates[branch][0]},
                mean_return=stats["mean_return"], mean_abs_finger_target_delta_mm=float(17.5*absolute.mean()),
                p95_abs_finger_target_delta_mm=float(17.5*np.quantile(absolute, .95)), worlds=worlds)
        entry["paired_counts"] = {
            f"{a}_to_{b}": paired_counts(all_gates[a], all_gates[b])
            for a, b in (("prior", "original_noise"), ("original_noise", "attenuated_fingers"), ("prior", "attenuated_fingers"))}
        report["groups"].append(entry)
        save_json(out / "results.json", report)
        print("training group", group, {b: entry["branches"][b]["counts"] for b in BRANCHES}, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new output directory for measured diagnostic results")
    run(parser.parse_args().out)

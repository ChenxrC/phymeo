"""Training-only paired comparison of the amplitude and persistence of pickup motor exploration."""
import argparse
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch

from _panda_pickup_rl_actor import PRIOR_SHA256, ResidualPickupActorCritic
from _panda_pickup_rl_domains import make_batch
from _panda_pickup_rl_native import run_native
from panda_pickup_exploration_check import CHECKPOINTS, DECISIONS, WORLDS, gates, paired_counts
from panda_pickup_from_reach import DT, REPEAT, apply_action
from panda_pickup_ppo import save_json


BATCH, SEED = 402, 271145
# Left-arm, right-arm and finger marginal raw standard deviations; arm AR coefficient.
LAWS = {
    "prior": (0., 0., 0., 0.),
    "batch401_thin_fingers_iid": (.035, .10, .04, 0.),
    "fine_iid": (.02, .02, .04, 0.), "fine_arm_ar": (.02, .02, .04, .8),
    "micro_iid": (.006, .006, .006, 0.), "micro_arm_ar": (.006, .006, .006, .8),
}
RULE = ["maximize total loaded pickups", "maximize minimum loaded pickups across the two domains",
        "maximize total worlds lifting 25 mm", "maximize total worlds with bilateral handle contact",
        "prefer IID", "prefer smaller sum of marginal raw variances"]
CANDIDATES = ("fine_iid", "fine_arm_ar", "micro_iid", "micro_arm_ar")


def latent_noise(draws, law):
    """Initialize from the stationary marginal. Only the fourteen arm motors retain past draws."""
    left, right, fingers, rho = law
    noise = draws * draws.new_tensor([left]*7 + [right]*7 + [fingers]*2)
    if rho:
        innovation = math.sqrt(1.-rho*rho)
        for t in range(1, len(noise)):
            noise[t, :, :14] = rho*noise[t-1, :, :14] + innovation*noise[t, :, :14]
    return noise


def select_law(groups):
    scores = {}
    for key in CANDIDATES:
        c = [group["branches"][key]["counts"] for group in groups]
        loaded = [v["loaded_pickup"] for v in c]
        left, right, finger, rho = LAWS[key]
        scores[key] = [sum(loaded), min(loaded), sum(v["lift"] for v in c), sum(v["bilateral_handle"] for v in c),
                       int(rho == 0.), -(7*left**2 + 7*right**2 + 2*finger**2)]
    best = max(CANDIDATES, key=scores.__getitem__)
    supported = scores[best][0] > 0
    selected = best if supported else "micro_iid"
    return dict(selected=selected, observed_loaded_success=supported, scores=scores, rule=RULE,
                fallback="If every eligible branch has zero loaded pickups, use micro_iid as the prior-near fallback.")


class ArmExplorationReplay(torch.nn.Module):
    def __init__(self, policy, domain, name, noise):
        super().__init__()
        self.policy, self.model, self.name, self.noise = policy, domain.model, name, noise
        self.targets = [h.target.copy() for h in domain.handoffs]
        self.prior_actions, self.deltas, self.target_deltas = [], [], []

    def observation(self, features, velocity):
        return self.policy.observation(features, velocity)

    def sample(self, obs, deterministic=False):
        if not deterministic:
            raise ValueError("Fixed-draw comparisons require the native candidate evaluation path")
        t = len(self.deltas)
        baseline = self.policy.prior_action(obs[:, :21])
        if self.name == "prior":
            action = baseline
        else:
            # No parameter updates: the freshly initialized residual Gaussian has identically zero mean.
            raw = self.noise[t]
            p = baseline.clamp(-1.+1e-6, 1.-1e-6)
            left, right, fingers, rho = LAWS[self.name]
            conditional_mean = torch.zeros_like(raw)
            std = raw.new_tensor([left]*7 + [right]*7 + [fingers]*2)
            if t and rho:
                conditional_mean[:, :14] = rho*self.noise[t-1, :, :14]
                std[:14] *= math.sqrt(1.-rho*rho)
            action, _ = self.policy._transport(p, raw, conditional_mean, std.log())
        base_np, actual_np = baseline.numpy(), action.numpy()
        # Include real joint bounds. Compare both commands at the branch's
        # current motor target, then advance the mirror with its issued action.
        measured = []
        for w, previous in enumerate(self.targets):
            nominal = apply_action(self.model, previous, base_np[w])
            actual = apply_action(self.model, previous, actual_np[w])
            measured.append(actual - nominal)
            self.targets[w] = actual
        self.prior_actions.append(base_np)
        self.deltas.append(actual_np - base_np)
        self.target_deltas.append(measured)
        return action, None, None, None

    def measurements(self):
        delta = np.stack(self.deltas)
        target = np.stack(self.target_deltas)
        prior = np.stack(self.prior_actions)
        near = np.abs(prior) >= .95
        # A negative radial change moves toward the center of the normalized action interval.
        signed = np.sign(prior[near])*delta[near]
        arms = self.noise[:, :, :14].numpy()
        duration = None
        if self.name != "prior":
            signs = np.sign(arms)
            runs = signs.shape[1]*signs.shape[2] + (signs[1:] != signs[:-1]).sum()
            duration = float(signs.size / runs * REPEAT * DT)
        return dict(mean_abs_arm_command_delta_rad=float(np.abs(target[:, :, :14]).mean()),
                    p95_abs_arm_command_delta_rad=float(np.quantile(np.abs(target[:, :, :14]), .95)),
                    mean_abs_finger_target_delta_mm=float(1000*np.abs(target[:, :, 14:]).mean()),
                    p95_abs_finger_target_delta_mm=float(1000*np.quantile(np.abs(target[:, :, 14:]), .95)),
                    realized_raw_std_by_motor=self.noise.numpy().std(axis=(0, 1)).tolist(),
                    mean_same_sign_arm_noise_duration_s=duration, near_bound_decisions_by_motor=near.sum((0, 1)).tolist(),
                    mean_outward_signed_action_delta_near_bound=float(signed.mean()) if signed.size else None)


def run(out):
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    handoffs = CHECKPOINTS / "panda-pickup-from-reach-s4000"
    xml = (CHECKPOINTS / "panda-reach-factors-s4000/scene.xml").read_text(encoding="utf-8")
    actor = ResidualPickupActorCritic.from_student(handoffs / "student.npz", "cpu").eval()
    draws = torch.randn((DECISIONS, WORLDS, 16), generator=torch.Generator().manual_seed(SEED))
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema="panda-pickup-arm-exploration-v1", split="train", batch=BATCH, seed=SEED, trained_updates=0,
                  worlds_per_group=WORLDS, decisions=DECISIONS, repeat=REPEAT, timestep_s=DT, prior_sha256=PRIOR_SHA256,
                  physical_duration_per_world_s=DECISIONS*REPEAT*DT,
                  draws_sha256=hashlib.sha256(draws.numpy().tobytes()).hexdigest(),
                  laws={k: dict(left_arm_sigma=l, right_arm_sigma=r, finger_sigma=f, arm_rho=rho)
                        for k, (l, r, f, rho) in LAWS.items()}, selection_rule=RULE,
                  eligible_candidates=list(CANDIDATES), no_loaded_pickups_fallback="micro_iid", groups=[])
    for group in (0, 1):
        domain = make_batch(xml, handoffs, "train", group, worlds=WORLDS, batch=BATCH)
        manifest = f"train-{group}-domain.json"
        save_json(out / manifest, domain.manifest)
        entry = dict(group=group, domain_file=manifest, resets_sha256=domain.manifest["resets_sha256"], branches={})
        by_branch = {}
        for name, law in LAWS.items():
            wrapper = ArmExplorationReplay(actor, domain, name, latent_noise(draws, law))
            started = time.monotonic()
            _, scores, stats = run_native(domain, wrapper, decisions=DECISIONS, repeat=REPEAT, mode="candidate")
            if len(scores) != WORLDS or len(wrapper.deltas) != DECISIONS:
                raise RuntimeError("Incomplete paired exploration evaluation")
            by_branch[name] = [gates(score) for score in scores]
            counts = {k: sum(row[k] for row in by_branch[name]) for k in by_branch[name][0]}
            entry["branches"][name] = dict(counts=counts, mean_return=stats["mean_return"], wall_seconds=time.monotonic()-started,
                                          measured=wrapper.measurements(), scores_in_manifest_order=scores)
            print("training group", group, name, counts, flush=True)
        comparisons = [("prior", name) for name in LAWS if name != "prior"]
        comparisons += [("fine_iid", "fine_arm_ar"), ("micro_iid", "micro_arm_ar"),
                        ("batch401_thin_fingers_iid", "fine_iid"), ("fine_iid", "micro_iid")]
        entry["paired_counts"] = {f"{a}_to_{b}": paired_counts(by_branch[a], by_branch[b]) for a, b in comparisons}
        report["groups"].append(entry)
        save_json(out / "results.json", report)
    report["selection"] = select_law(report["groups"])
    save_json(out / "results.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for training-domain comparisons")
    run(parser.parse_args().out)

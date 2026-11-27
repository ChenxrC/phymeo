"""Residual pickup PPO on the published 21-feature RBF motor policy."""
import copy
import hashlib
import io
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from _panda_pickup_student import PickupStudent


PRIOR_SHA256 = "4c8f50d08495017c7bd52715ccbe913e64fde9bbcfffde8ad453914dae8a835a"
SCHEMA = "dual-panda-pickup-residual-ppo-v1"
EXPLORATION_SCHEMA = "dual-panda-pickup-residual-ppo-v2"
CONTRACT = dict(
    prior_sha256=PRIOR_SHA256, features="published pickup-v1 21 values followed by native qvel[0:22]",
    velocity_scale=[2.] * 14 + [.20] * 2 + [.30] * 3 + [2.] * 3,
    transport="p=RBF clipped to ±(1-1e-6); action=p+tanh(raw)*(1-sign(raw)*p), positive branch at zero",
    policy="diagonal Gaussian residual, corrected action log density, Gaussian entropy regularizer",
    dimensions=[43, 64, 64, 16])


def _rotvec(t, r):
    """Principal SO(3) logarithm using the largest quaternion component, including near pi."""
    a, b, c = r[..., 0, :].unbind(-1)
    d, e, f = r[..., 1, :].unbind(-1)
    g, h, i = r[..., 2, :].unbind(-1)
    squares = t.stack((1+a+e+i, 1+a-e-i, 1-a+e-i, 1-a-e+i), -1)
    candidates = t.stack((t.stack((squares[..., 0], h-f, c-g, d-b), -1),
                          t.stack((h-f, squares[..., 1], b+d, c+g), -1),
                          t.stack((c-g, b+d, squares[..., 2], f+h), -1),
                          t.stack((d-b, c+g, f+h, squares[..., 3]), -1)), -2)
    index = squares.argmax(-1)
    quat = candidates.gather(-2, index[..., None, None].expand(*index.shape, 1, 4)).squeeze(-2)
    quat = quat / quat.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
    quat = t.where(quat[..., :1] < 0, -quat, quat)
    vec = quat[..., 1:]
    sine = vec.square().sum(-1, keepdim=True).sqrt()
    angle = 2*t.atan2(sine, quat[..., :1])
    return vec * t.where(sine > 1e-7, angle / sine.clamp_min(1e-12), t.full_like(sine, 2.))


def pickup_features(torch, poses, q, target, initial_pin_z):
    """Recreate PickupFeatures.observe from ReachTask.poses and the pickup handoff pin height."""
    _, (hand, rh), (pin, rp) = poses
    translation = (rh * (pin-hand).unsqueeze(-1)).sum(-2)
    rotation = (rh.transpose(-1, -2).unsqueeze(-1) * rp.unsqueeze(-3)).sum(-2)
    z0 = torch.as_tensor(initial_pin_z, device=q.device, dtype=q.dtype).reshape(-1)
    return torch.cat((translation, _rotvec(torch, rotation), rp[:, :2, 0],
                      (pin[:, 2]-z0)[:, None], (hand[:, 2]-z0)[:, None],
                      q[:, 14:16], target[:, 14:16], target[:, 7:14]-q[:, 7:14]), -1)


def _network(output):
    return nn.Sequential(nn.Linear(43, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, output))


class ResidualPickupActorCritic(nn.Module):
    """All buffers, policy operations and value estimation follow the selected device."""
    def __init__(self, student, *, arm_std=None, finger_std=None, arm_rho=0.):
        super().__init__()
        if student.mean.shape != (21,):
            raise ValueError("Expected the 21-feature pickup RBF")
        if type(arm_rho) not in (int, float) or arm_rho not in (0., .8):
            raise ValueError("arm_rho must be 0 or 0.8")
        self.arm_rho = float(arm_rho)
        self.exploration = None
        self.log_std_min = -5.
        if arm_std is not None or finger_std is not None or self.arm_rho:
            self.log_std_min = -8.
            for value in (arm_std, finger_std):
                if value is not None and (type(value) not in (int, float) or
                                          not math.exp(self.log_std_min) <= value <= math.exp(.5)):
                    raise ValueError("starting std must be finite and within the exploration scale bounds")
            self.exploration = dict(
                arm_std=arm_std, finger_std=finger_std, arm_rho=self.arm_rho,
                initial_sigma=([arm_std]*14 if arm_std is not None else [.035]*7+[.10]*7)
                              + [finger_std if finger_std is not None else .20]*2,
                log_std_bounds=[self.log_std_min, .5], learn_log_std=True,
                first_draw="N(mu(obs), sigma^2) independently per joint",
                following_arms="N((1-rho)*mu(obs)+rho*previous_raw, (1-rho^2)*sigma^2)",
                following_fingers="N(mu(obs), sigma^2) independently",
                evaluation="conditional mean recursively; reset history at episode start")
        for name in ("mean", "inverse_scale", "centers", "weights"):
            self.register_buffer("prior_" + name, torch.tensor(getattr(student, name), dtype=torch.float32),
                                 persistent=False)
        self.register_buffer("velocity_scale", torch.tensor(CONTRACT["velocity_scale"], dtype=torch.float32),
                             persistent=False)
        self.actor = _network(16)
        self.critic = _network(1)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        # Raw residual amplitude before the headroom transform. Both jaws
        # explore independently even when the prior commands fully open.
        initial_sigma = self.exploration["initial_sigma"] if self.exploration else [.035]*7 + [.10]*7 + [.20]*2
        self.log_std = nn.Parameter(torch.tensor(initial_sigma, dtype=torch.float32).log())

    @classmethod
    def from_student(cls, path, device, *, arm_std=None, finger_std=None, arm_rho=0.):
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != PRIOR_SHA256:
            raise ValueError("Pickup PPO requires the published frozen RBF weights")
        return cls(PickupStudent.load(io.BytesIO(payload)), arm_std=arm_std, finger_std=finger_std,
                   arm_rho=arm_rho).to(device)

    def observation(self, features21, qvel22):
        x, v = (torch.as_tensor(a, device=self.prior_mean.device, dtype=self.prior_mean.dtype)
                for a in (features21, qvel22))
        if x.shape[-1] != 21 or v.shape[-1] != 22 or x.shape[:-1] != v.shape[:-1]:
            raise ValueError("Expected matching 21-feature pickup observations and 22 native velocities")
        return torch.cat((x, v), -1)

    def _standardize(self, x):
        return ((x - self.prior_mean) * self.prior_inverse_scale).clamp(-8., 8.)

    def _rbf(self, z):
        # Direct differences and a weighted sum keep the frozen FP32 prior
        # independent of reduced-precision matrix-multiplication settings.
        d = (z.unsqueeze(-2) - self.prior_centers).square().sum(-1)
        nearest = torch.topk(d, k=min(6, len(self.prior_centers)), dim=-1, largest=False, sorted=True).values
        logits = -(d - nearest[..., :1]) / (.25 * nearest[..., -1:]).clamp_min(1e-10)
        phi = torch.softmax(logits, dim=-1)
        return (phi.unsqueeze(-1) * self.prior_weights).sum(-2).clamp(-1., 1.)

    def prior_action(self, features21):
        x = torch.as_tensor(features21, device=self.prior_mean.device, dtype=self.prior_mean.dtype)
        return self._rbf(self._standardize(x))

    def _inputs(self, obs):
        if obs.shape[-1] != 43:
            raise ValueError("Expected a packed [pickup21, native qvel22] observation")
        z = self._standardize(obs[..., :21])
        x = torch.cat((z, (obs[..., 21:] / self.velocity_scale).clamp(-8., 8.)), -1)
        return z, x

    def value(self, obs):
        _, x = self._inputs(obs)
        return self.critic(x).squeeze(-1)

    def _distribution(self, obs, prev_raw=None, has_prev=None):
        z, x = self._inputs(obs)
        prior = self._rbf(z).clamp(-1.+1e-6, 1.-1e-6)
        mean = self.actor(x)
        log_std = self.log_std.clamp(self.log_std_min, .5)
        if (prev_raw is None) != (has_prev is None):
            raise ValueError("supply both previous raw residual and history mask")
        if self.arm_rho and prev_raw is not None:
            if (prev_raw.shape != mean.shape or has_prev.shape != mean.shape[:-1]
                    or prev_raw.dtype != mean.dtype or has_prev.dtype != torch.bool
                    or prev_raw.device != mean.device or has_prev.device != mean.device):
                raise ValueError("expected matching previous raw residuals and boolean history mask")
            mask = has_prev.unsqueeze(-1)
            previous = prev_raw[..., :14].detach()
            arm_mean = torch.where(mask, (1.-self.arm_rho)*mean[..., :14] + self.arm_rho*previous, mean[..., :14])
            mean = torch.cat((arm_mean, mean[..., 14:]), dim=-1)
            arm_scale = log_std[:14] + mask.to(mean.dtype) * (.5*math.log1p(-self.arm_rho**2))
            log_std = torch.cat((arm_scale, log_std[14:].expand_as(mean[..., 14:])), dim=-1)
        return prior, mean, log_std, self.critic(x).squeeze(-1)

    @staticmethod
    def _transport(prior, raw, mean, log_std):
        scale = torch.where(raw >= 0., 1.-prior, 1.+prior)
        action = (prior + scale * raw.tanh()).clamp(-1., 1.)
        base = -.5*((raw-mean)*(-log_std).exp()).square() - log_std - .5*math.log(2.*math.pi)
        log_jacobian = scale.log() + 2.*(math.log(2.)-raw-F.softplus(-2.*raw))
        return action, (base-log_jacobian).sum(-1)

    def sample(self, obs, deterministic=False, *, prev_raw=None, has_prev=None):
        prior, mean, log_std, value = self._distribution(obs, prev_raw, has_prev)
        raw = mean if deterministic else mean + log_std.exp()*torch.randn_like(mean)
        action, logp = self._transport(prior, raw, mean, log_std)
        return action, raw, logp, value

    def evaluate(self, obs, raw, *, prev_raw=None, has_prev=None):
        if self.arm_rho and prev_raw is None:
            raise ValueError("correlated PPO evaluation requires stored previous_raw and has_prev conditioning")
        prior, mean, log_std, value = self._distribution(obs, prev_raw, has_prev)
        _, logp = self._transport(prior, raw.detach(), mean, log_std)
        gaussian_entropy = (log_std + .5*math.log(2.*math.pi*math.e)).sum(dim=-1).expand_as(value)
        return logp, gaussian_entropy, value

    def checkpoint(self):
        contract = copy.deepcopy(CONTRACT)
        if self.exploration:
            contract["exploration"] = copy.deepcopy(self.exploration)
        return dict(schema=EXPLORATION_SCHEMA if self.exploration else SCHEMA, contract=contract,
                    state={k: v.detach().cpu().clone() for k, v in self.state_dict().items()})

    @classmethod
    def from_checkpoint(cls, payload, student_path, device):
        if not isinstance(payload.get("contract"), dict):
            raise ValueError("Pickup PPO checkpoint contract must be a mapping")
        contract = payload["contract"].copy()
        has_exploration = "exploration" in contract
        config = contract.pop("exploration", None)
        if (contract != CONTRACT or payload.get("schema") != (EXPLORATION_SCHEMA if has_exploration else SCHEMA)
                or (has_exploration and (not isinstance(config, dict)
                                            or not {"arm_std", "finger_std", "arm_rho"} <= config.keys()))):
            raise ValueError("Pickup PPO checkpoint observation, prior or action contract differs")
        args = {k: config[k] for k in ("arm_std", "finger_std", "arm_rho")} if config is not None else {}
        policy = cls.from_student(student_path, device, **args)
        if config != policy.exploration:
            raise ValueError("Pickup PPO checkpoint exploration contract differs")
        template = policy.state_dict()
        state = payload["state"]
        if state.keys() != template.keys() or any(
                not isinstance(v, torch.Tensor) or v.shape != template[k].shape or v.dtype != template[k].dtype
                or not torch.isfinite(v).all().item() for k, v in state.items()):
            raise ValueError("Pickup PPO checkpoint state differs from the declared architecture")
        policy.load_state_dict(state, strict=True)
        return policy


def rollout_action(policy, obs, mode, records, previous):
    """Select physical motor actions and record the same conditioning later used by PPO."""
    if mode == "prior":
        return policy.prior_action(obs[:, :21]), None
    history = {}
    if policy.arm_rho:
        history["prev_raw"] = obs.new_zeros((len(obs), 16)) if previous is None else previous.detach()
        history["has_prev"] = torch.full((len(obs),), previous is not None, dtype=torch.bool, device=obs.device)
    action, raw, logp, value = policy.sample(obs, deterministic=mode == "candidate", **history)
    if mode == "train":
        for k, x in dict(obs=obs, raw=raw, logp=logp, value=value, **history).items():
            records.setdefault(k, []).append(x.detach())
    return action, raw.detach()

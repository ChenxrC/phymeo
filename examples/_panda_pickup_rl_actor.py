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
    def __init__(self, student):
        super().__init__()
        if student.mean.shape != (21,):
            raise ValueError("Expected the 21-feature pickup RBF")
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
        self.log_std = nn.Parameter(torch.tensor([.035]*7 + [.10]*7 + [.20]*2).log())

    @classmethod
    def from_student(cls, path, device):
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != PRIOR_SHA256:
            raise ValueError("Pickup PPO requires the published frozen RBF weights")
        return cls(PickupStudent.load(io.BytesIO(payload))).to(device)

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

    def _distribution(self, obs):
        z, x = self._inputs(obs)
        prior = self._rbf(z).clamp(-1.+1e-6, 1.-1e-6)
        return prior, self.actor(x), self.log_std.clamp(-5., .5), self.critic(x).squeeze(-1)

    @staticmethod
    def _transport(prior, raw, mean, log_std):
        scale = torch.where(raw >= 0., 1.-prior, 1.+prior)
        action = (prior + scale * raw.tanh()).clamp(-1., 1.)
        base = -.5*((raw-mean)*(-log_std).exp()).square() - log_std - .5*math.log(2.*math.pi)
        log_jacobian = scale.log() + 2.*(math.log(2.)-raw-F.softplus(-2.*raw))
        return action, (base-log_jacobian).sum(-1)

    def sample(self, obs, deterministic=False):
        prior, mean, log_std, value = self._distribution(obs)
        raw = mean if deterministic else mean + log_std.exp()*torch.randn_like(mean)
        action, logp = self._transport(prior, raw, mean, log_std)
        return action, raw, logp, value

    def evaluate(self, obs, raw):
        prior, mean, log_std, value = self._distribution(obs)
        _, logp = self._transport(prior, raw.detach(), mean, log_std)
        gaussian_entropy = (log_std + .5*math.log(2.*math.pi*math.e)).sum().expand_as(value)
        return logp, gaussian_entropy, value

    def checkpoint(self):
        return dict(schema=SCHEMA, contract=copy.deepcopy(CONTRACT),
                    state={k: v.detach().cpu().clone() for k, v in self.state_dict().items()})

    @classmethod
    def from_checkpoint(cls, payload, student_path, device):
        if payload.get("schema") != SCHEMA or payload.get("contract") != CONTRACT:
            raise ValueError("Pickup PPO checkpoint observation, prior or action contract differs")
        policy = cls.from_student(student_path, device)
        template = policy.state_dict()
        state = payload["state"]
        if state.keys() != template.keys() or any(
                not isinstance(v, torch.Tensor) or v.shape != template[k].shape or v.dtype != template[k].dtype
                or not torch.isfinite(v).all().item() for k, v in state.items()):
            raise ValueError("Pickup PPO checkpoint state differs from the declared architecture")
        policy.load_state_dict(state, strict=True)
        return policy

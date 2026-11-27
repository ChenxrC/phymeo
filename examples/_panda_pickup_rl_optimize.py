"""Finite-episode PPO updates for the residual Panda pickup actor and value network."""
import math

from panda_state_ppo import generalized_advantage


def update_policy(torch, actor, optimizer, records, *, epochs=3, minibatch=256, clip=.15, entropy_weight=.001):
    """Update on one [decision, world] batch ending at a genuine terminal state in every world.

    The rollout stores the policy's raw residual draws and transported motor log densities.
    All training arithmetic stays on the actor's device. Returned metrics contain CPU scalars.
    """
    if (type(epochs) is not int or epochs < 1 or type(minibatch) is not int or minibatch < 1
            or not 0. < clip < 1. or not math.isfinite(entropy_weight) or entropy_weight < 0.):
        raise ValueError("expected positive integer epochs/minibatch, clip in (0,1), finite nonnegative entropy weight")
    obs, raw, old_logp, values, rewards = (records[k].detach() for k in ("obs", "raw", "logp", "value", "reward"))
    if (rewards.ndim != 2 or min(rewards.shape) < 1 or values.shape != rewards.shape or old_logp.shape != rewards.shape
            or obs.shape != (*rewards.shape, 43) or raw.shape != (*rewards.shape, 16)):
        raise ValueError("expected obs[T,W,43], raw[T,W,16], and scalar logp/value/reward[T,W]")
    parameter = next(actor.parameters())
    arrays = (obs, raw, old_logp, values, rewards)
    if any(x.device != parameter.device or x.dtype != parameter.dtype for x in arrays):
        raise ValueError("all rollout tensors must share the residual actor's device and dtype")
    if not bool(torch.stack([torch.isfinite(x).all() for x in arrays]).all()):
        raise ValueError("nonfinite PPO rollout")

    gamma, lam = .99, .95
    with torch.no_grad():
        adv, returns = generalized_advantage(torch, rewards, values, torch.zeros_like(values[-1]), gamma, lam)
        adv, returns = adv.flatten(), returns.flatten()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        if not bool(torch.isfinite(adv).all() & torch.isfinite(returns).all()):
            raise ValueError("nonfinite terminal GAE")
    obs, raw, old_logp = obs.flatten(0, 1), raw.flatten(0, 1), old_logp.flatten()
    n = len(adv)
    trainable = {k: p for k, p in actor.named_parameters() if p.requires_grad}
    saved = {k: p.detach().clone() for k, p in trainable.items()}
    # Weight per-example losses by minibatch size; gradient norms describe optimizer steps.
    totals = torch.zeros(5, device=parameter.device, dtype=parameter.dtype)
    norms = []
    for _ in range(epochs):
        perm = torch.randperm(n, device=parameter.device)
        for ids in perm.split(minibatch):
            logp, entropy, value = actor.evaluate(obs[ids], raw[ids])
            log_ratio = logp - old_logp[ids]
            ratio = log_ratio.exp()
            bounded = ratio.clamp(1. - clip, 1. + clip)
            policy_loss = -torch.minimum(ratio * adv[ids], bounded * adv[ids]).mean()
            value_loss = (value - returns[ids]).square().mean()
            entropy = entropy.mean()
            loss = policy_loss + .5 * value_loss - entropy_weight * entropy
            if not bool(torch.isfinite(loss) & torch.isfinite(ratio).all()):
                raise ValueError("nonfinite PPO minibatch loss or likelihood ratio")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                kl = (ratio - 1. - log_ratio).clamp_min(0.).mean()
                fraction = ((ratio - 1.).abs() > clip).to(ratio.dtype).mean()
                totals += len(ids) * torch.stack((policy_loss, value_loss, entropy, kl, fraction))
                norms.append(norm.detach())

    with torch.no_grad():
        # A single-minibatch update starts with ratio=1. Evaluate the resulting policy as well.
        final = torch.zeros_like(totals[:2])
        for start in range(0, n, minibatch):
            batch = slice(start, min(start + minibatch, n))
            logp, _, _ = actor.evaluate(obs[batch], raw[batch])
            log_ratio = logp - old_logp[batch]
            ratio = log_ratio.exp()
            final[0] += (ratio - 1. - log_ratio).clamp_min(0.).sum()
            final[1] += ((ratio - 1.).abs() > clip).sum()
        changes = {name: (weight - saved[name]).abs().max() for name, weight in trainable.items()}
        def largest(prefix):
            return torch.stack([delta for name, delta in changes.items() if name.startswith(prefix)]).max()
        norm = torch.stack(norms)
        names = ("mean_policy_loss", "mean_value_loss", "mean_entropy", "training_approx_kl", "training_clip_fraction",
                 "approx_kl", "clip_fraction", "mean_grad_norm", "max_grad_norm", "max_parameter_change",
                 "actor_max_parameter_change", "critic_max_parameter_change", "log_std_max_change", "mean_gae_return")
        metrics = torch.cat((totals / (n * epochs), final / n, torch.stack((norm.mean(), norm.max(), largest(""),
                             largest("actor."), largest("critic."), changes["log_std"], returns.mean()))))
        if not bool(torch.isfinite(metrics).all()):
            raise ValueError("nonfinite updated policy or PPO metrics")
        return dict(samples=n, sample_visits=n * epochs, optimizer_steps=len(norms), gamma=gamma, gae_lambda=lam,
                    terminal_bootstrap=0., **dict(zip(names, metrics.cpu().tolist(), strict=True)))

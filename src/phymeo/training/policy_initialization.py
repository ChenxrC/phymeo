"""Optional one-shot mean-head initialization for a bounded RSL5.5.2 MLP domain.

No physics, optimizer update, policy loop or task identity is embedded here.
Only the actual mean head bias may be published after candidate validation.
"""

import copy
import hashlib
import json
import random
from importlib.metadata import version
from types import SimpleNamespace

import numpy as np

from . import mujoco_tyx_evaluation as evidence
from .policy_response_diagnostics import _preserved_execution_state


class InitializationFailure(RuntimeError):
    def __init__(self, error, *, rolled_back, receipt):
        super().__init__("initial mean-centering failed: " + str(error))
        self.rolled_back = rolled_back
        self.receipt = receipt


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(evidence.encode(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _model_state(model):
    result = {}
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous().numpy()
        result[name] = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    return result


def _rng_bindings(torch):
    return {
        "python": _digest(random.getstate()), "numpy": _digest(np.random.get_state()),
        "torch_cpu": _digest(torch.get_rng_state()),
        "torch_cuda": [_digest(value) for value in torch.cuda.get_rng_state_all()],
    }


def _mean(actor, observation):
    return actor(observation, stochastic_output=False)


def _publish_bias(bias, value):
    import torch
    with torch.no_grad():
        bias.copy_(value)


def _synchronize(device, torch):
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


def _verify_published(actor, critic, optimizer, expected_hash, optimizer_hash):
    actual = evidence.parameter_hash(SimpleNamespace(actor=actor, critic=critic))
    _require(actual == expected_hash, "actual published model differs from candidate")
    _require(_digest(optimizer.state_dict()) == optimizer_hash, "optimizer changed during initialization")
    return actual


def _validate(actor, critic, optimizer, observation, expected_policy_hash, torch):
    from rsl_rl.models import MLPModel
    from rsl_rl.modules import MLP
    from rsl_rl.modules.distribution import GaussianDistribution
    from tensordict import TensorDict

    _require(version("rsl-rl-lib") == "5.5.2", "exact RSL5.5.2 required")
    _require(type(actor) is MLPModel and type(critic) is MLPModel, "exact upstream MLP actor/critic")
    _require(not actor.is_recurrent and not critic.is_recurrent, "feedforward initialization only")
    _require(actor.obs_normalization is False and type(actor.obs_normalizer) is torch.nn.Identity,
             "actor observation normalization must be Identity")
    _require(type(actor.mlp) is MLP and type(actor.distribution) is GaussianDistribution,
             "only the upstream MLP with state-independent Gaussian mean")
    _require(actor.distribution._distribution is None, "fresh Gaussian cache required")
    layers = tuple(actor.mlp.children())
    _require(1 <= len(layers) <= 33 and len(layers) % 2 == 1, "bounded alternating MLP layout")
    _require(all(type(layer) is (torch.nn.Linear if index % 2 == 0 else torch.nn.Tanh)
                 for index, layer in enumerate(layers)), "Linear/Tanh whitelist")
    head = layers[-1]
    _require(head.bias is not None and 1 <= head.out_features <= 35, "actual bounded mean head bias")
    _require(type(optimizer) is torch.optim.Adam and not optimizer.state, "fresh Adam required")
    parameters = list(actor.parameters()) + list(critic.parameters())
    identities = [id(value) for value in parameters]
    actual_ids = [id(value) for group in optimizer.param_groups for value in group["params"]]
    _require(len(set(identities)) == len(identities)
             and sorted(actual_ids) == sorted(identities), "exact unique actor/critic Adam ownership")
    device = head.bias.device
    _require(device.type in ("cpu", "cuda"), "CPU/CUDA model required")
    _require(sum(value.numel() for value in parameters) <= 1_048_576, "model scalar ceiling")
    _require(all(value.dtype == torch.float32 and value.device == device and value.grad is None
                 and bool(torch.isfinite(value).all()) for value in parameters),
             "finite FP32 same-device parameters with no pending gradients")
    _require(type(observation) is TensorDict and tuple(observation.batch_size) == (1,),
             "one explicit TensorDict reference observation")
    _require(set(observation.keys()) == set(actor.obs_groups), "exact actor observation keys")
    total_width = 0
    for name in actor.obs_groups:
        value = observation[name]
        _require(value.dtype == torch.float32 and value.device == device and value.ndim == 2
                 and value.shape[0] == 1 and bool(torch.isfinite(value).all()), "reference observation layout/finite")
        total_width += value.shape[1]
    _require(1 <= total_width <= 256 and total_width == actor.obs_dim, "bounded complete feature layout")
    _require(type(expected_policy_hash) is str and len(expected_policy_hash) == 64
             and evidence.parameter_hash(SimpleNamespace(actor=actor, critic=critic)) == expected_policy_hash,
             "original complete actor/critic hash")
    bias_name = next(name for name, value in actor.named_parameters() if value is head.bias)
    return head.bias, bias_name, device


def center_initial_mean(actor, observation, *, critic, optimizer, expected_policy_hash):
    """Center one initial reference mean; after return the bias trains normally.

    Requires exclusive ownership and an empty fresh Adam. Only one actual model
    tensor is copied. Any failure after that copy attempts exact bias rollback.
    """
    import torch

    bias, bias_name, device = _validate(
        actor, critic, optimizer, observation, expected_policy_hash, torch)
    before = {"actor": _model_state(actor), "critic": _model_state(critic)}
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    optimizer_hash = _digest(optimizer_before)
    original_bias = bias.detach().clone()
    rng_before = _rng_bindings(torch)
    receipt = {"schema": "rsl-mlp-one-shot-mean-centering-v1",
               "status": "PREPARED_NOT_PUBLISHED", "bias_name": bias_name,
               "model_hash_before": expected_policy_hash, "model_before": before,
               "optimizer_before": optimizer_before, "rng_before": rng_before}
    publication_started = False
    try:
        with _preserved_execution_state(torch) as flags:
            with torch.no_grad():
                mean_before = _mean(actor, observation).detach().clone()
                _require(tuple(mean_before.shape) == (1, bias.numel())
                         and mean_before.dtype == torch.float32 and mean_before.device == device
                         and bool(torch.isfinite(mean_before).all()), "finite original mean layout")
                proposed_bias = original_bias - mean_before[0]
                _require(bool(torch.isfinite(proposed_bias).all()), "finite FP32 bias subtraction")
                candidate = copy.deepcopy(actor)
                candidate_bias = dict(candidate.named_parameters())[bias_name]
                candidate_bias.copy_(proposed_bias)
                mean_after = _mean(candidate, observation).detach().clone()
                _require(bool(torch.isfinite(mean_after).all()), "finite candidate residual")
                candidate_state = _model_state(candidate)
                candidate_hash = evidence.parameter_hash(SimpleNamespace(actor=candidate, critic=critic))
                for name, value in candidate_state.items():
                    if name != bias_name:
                        _require(np.array_equal(value, before["actor"][name]), "candidate changed a non-bias tensor")
                _synchronize(device, torch)
                receipt.update(
                    model_after={"actor": candidate_state, "critic": before["critic"]},
                    candidate_model_hash=candidate_hash,
                    reference_observation={name: value.detach().cpu().numpy().copy()
                                           for name, value in observation.items()},
                    mean_before=mean_before.cpu().numpy().copy(),
                    mean_after=mean_after.cpu().numpy().copy(),
                    proposed_bias=proposed_bias.cpu().numpy().copy(),
                    computation_flags=dict(flags),
                )
        # RNG/config restoration is complete before the sole actual mutation.
        rng_after = _rng_bindings(torch)
        _require(rng_after == rng_before, "caller RNG changed during candidate preparation")
        _require(evidence.parameter_hash(SimpleNamespace(actor=actor, critic=critic)) == expected_policy_hash,
                 "original model changed before publication")
        _require(_digest(optimizer.state_dict()) == optimizer_hash, "Adam changed before publication")
        publication_started = True
        _publish_bias(bias, proposed_bias)
        _synchronize(device, torch)
        actual = _verify_published(actor, critic, optimizer, candidate_hash, optimizer_hash)
        receipt.update(status="APPLIED_ONE_SHOT_MEAN_CENTERING", model_hash_after=actual,
                       optimizer_after=copy.deepcopy(optimizer.state_dict()), rng_after=rng_after,
                       future_bias="ordinary trainable parameter; no per-step subtraction",
                       residual_gate="reported only; no exact-zero gate or repeated correction",
                       physics_steps=0, optimizer_steps=0)
        return receipt
    except BaseException as error:
        rolled_back = False
        if publication_started:
            try:
                with torch.no_grad():
                    bias.copy_(original_bias)
                _synchronize(device, torch)
                _require(evidence.parameter_hash(SimpleNamespace(actor=actor, critic=critic)) == expected_policy_hash,
                         "bias rollback did not restore the original model")
                rolled_back = True
            except BaseException as rollback_error:
                raise BaseExceptionGroup("initialization and bias rollback both failed",
                                         [error, rollback_error]) from error
        if isinstance(error, Exception):
            raise InitializationFailure(error, rolled_back=rolled_back, receipt=receipt) from error
        raise

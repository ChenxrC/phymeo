"""Short CUDA PPO qualification on two unchanged N1 native models.

Four updates are an interface/optimizer qualification, not evidence of a solved
task. Public task exports supply observations; HOST copies below are diagnostics.
"""
import argparse
import hashlib
import importlib.util
import json
import traceback
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

SEEDS = {"hinge-slide": 11011, "support-slide": 11012}
HORIZONS = (5, 7, 9, 11, 13, 15, 17, 19)
UPDATES, STEPS = 4, 16


def load_n1():
    """Reuse the accepted sibling fixture/example; never import tests."""
    path = Path(__file__).with_name("mujoco_tyx_cuda_policy.py")
    spec = importlib.util.spec_from_file_location("phymeo_n1_public_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_cuda():
    return load_n1().require_cuda()


@contextmanager
def make_environment(case, *, count=8, horizons=HORIZONS, task_spec=None):
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxActionSpec, MuJoCoTyxVectorEnv

    if case not in SEEDS or type(count) is not int or count != 8:
        raise ValueError("N2 qualification is frozen to two W8 tasks")
    n1 = load_n1()
    n1.require_cuda()
    source = n1.fixture_source(case)
    model = source.fresh_native_model()
    keys = n1.world_keys(count)
    spec = n1.task_spec(model) if task_spec is None else task_spec
    action_spec = MuJoCoTyxActionSpec(
        tuple(range(model.nu)), tuple(range(model.nv)),
        tuple((model.nbody - 1, component) for component in range(6)),
        scale=[0.02] * model.nu + [0.01] * (model.nv + 6),
        bias=[0.01] * model.nu + [0.005] * model.nv + [0.001 * (i + 1) for i in range(6)],
    )
    with ExitStack() as owners:
        runtime = owners.enter_context(MuJoCoRuntime(
            source, mode="cuda", world_keys=keys,
            initial_states=n1.initial_states(case, model, keys),
        ))
        env = owners.enter_context(MuJoCoTyxVectorEnv(
            runtime, spec, action_spec, h=n1.H[case], horizons=horizons,
        ))
        yield env


def diagnostic(value):
    """Explicit HOST evidence conversion, never a policy observation source."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    if is_dataclass(value):
        return {"type": type(value).__module__ + "." + type(value).__name__,
                "fields": {field.name: diagnostic(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, Mapping):
        return {str(key): diagnostic(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostic(item) for item in value]
    return value


def transition_record(transition):
    return {field.name: diagnostic(getattr(transition, field.name)) for field in fields(transition)}


def parameter_snapshot(algorithm):
    arrays = {
        owner + "." + name: value.detach().cpu().numpy().copy()
        for owner, network in (("actor", algorithm.actor), ("critic", algorithm.critic))
        for name, value in network.state_dict().items()
    }
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        digest.update(name.encode() + array.dtype.str.encode())
        digest.update(str(array.shape).encode() + array.tobytes())
    return digest.hexdigest(), arrays


def _setup(torch, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _configuration(case, seed, env, algorithm, recurrent):
    from importlib.metadata import version
    from phymeo.training.rsl_mujoco_tyx import ppo_config

    n1 = load_n1()
    return {
        "event": "configuration", "case": case, "seed": seed,
        "world_keys": env.world_keys, "h": env.h, "horizons": env.horizons,
        "xml": n1.XML[case], "xml_sha256": hashlib.sha256(n1.XML[case].encode()).hexdigest(),
        "binding": diagnostic(env.binding),
        "task": diagnostic(env.task_spec), "action": diagnostic(env.action_spec),
        "config": ppo_config(steps=8 if recurrent else STEPS, recurrent=recurrent),
        "upstream": {"rsl": version("rsl-rl-lib"), "torch": version("torch"),
                     "tensordict": version("tensordict")},
        "actor": str(algorithm.actor), "critic": str(algorithm.critic),
        "initial_parameters": parameter_snapshot(algorithm)[1],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "short-PPO-protocol-not-task-learning; physical-HOST-audit-retained",
        "runtime_caps": env.runtime.capabilities,
    }


def _failure(log, n1, phase, error, env):
    data = {
        "event": "failure", "phase": phase, "type": type(error).__name__,
        "message": str(error), "traceback": traceback.format_exc(),
    }
    failure = None if env is None else env.last_failure
    if failure is not None:
        data["wrapper_failure"] = {
            name: diagnostic(getattr(failure, name)) for name in
            ("phase", "action_accepted", "state_accepted", "reset_accepted",
             "action_receipt", "accepted_worlds", "needs_recovery")
        }
    if env is not None:
        data["last_successful_transition"] = (
            None if env.last_transition is None else transition_record(env.last_transition)
        )
    n1.append(log, data)


def train_case(case, output):
    """Original upstream PPO loop, fixed four updates; save complete transitions."""
    from phymeo.training.rsl_mujoco_tyx import (
        RslMuJoCoTyxVecEnv, make_ppo, optimization_evidence,
    )

    if case not in SEEDS:
        raise ValueError("unknown frozen N2 case")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch, n1 = require_cuda(), load_n1()
    seed = SEEDS[case]
    _setup(torch, seed)
    env = None
    phase = "construction"
    with output.open("x", encoding="utf-8") as log:
        try:
            with make_environment(case) as env:
                adapter = RslMuJoCoTyxVecEnv(env)
                algorithm = make_ppo(adapter, steps=STEPS)
                algorithm.train_mode()
                n1.append(log, _configuration(case, seed, env, algorithm, False))
                initial_hash, _ = parameter_snapshot(algorithm)
                observation = adapter.get_observations()
                optimizer_steps, losses, ends = [], [], np.zeros(8, np.int64)
                for update in range(UPDATES):
                    before_hash, before_parameters = parameter_snapshot(algorithm)
                    with torch.inference_mode():
                        for step in range(STEPS):
                            phase = "policy-act"
                            action = algorithm.act(observation)
                            entry_values = algorithm.transition.values.clone()
                            phase = "device-environment"
                            observation, reward, done, extras = adapter.step(action)
                            transition = adapter.last_transition
                            ends += transition.final_mask.cpu().numpy()
                            phase = "upstream-storage"
                            algorithm.process_env_step(observation, reward, done, extras)
                            if algorithm.storage.step != step + 1:
                                raise RuntimeError("upstream storage did not consume exactly one transition")
                            n1.append(log, {
                                "event": "transition", "update": update, "rollout_step": step,
                                **transition_record(transition),
                                "entry_values": diagnostic(entry_values),
                                "stored_reward": diagnostic(algorithm.storage.rewards[step]),
                                "stored_done": diagnostic(algorithm.storage.dones[step]),
                                "storage_step": algorithm.storage.step,
                            })
                        phase = "upstream-returns"
                        algorithm.compute_returns(observation)
                        n1.append(log, {
                            "event": "returns", "update": update,
                            "values": diagnostic(algorithm.storage.values),
                            "returns": diagnostic(algorithm.storage.returns),
                            "advantages": diagnostic(algorithm.storage.advantages),
                        })
                    phase = "upstream-backward-and-optimizer"
                    loss = algorithm.update()
                    if not all(np.isfinite(value) for value in loss.values()):
                        raise RuntimeError("nonfinite upstream loss")
                    evidence = optimization_evidence(algorithm)
                    if evidence["optimizer_step"] != update + 1 or algorithm.storage.step != 0:
                        raise RuntimeError("wrong upstream optimizer count/storage-clear boundary")
                    after_hash, after_parameters = parameter_snapshot(algorithm)
                    if before_hash == after_hash:
                        raise RuntimeError("upstream update did not change parameters")
                    optimizer_steps.append(evidence["optimizer_step"])
                    losses.append(loss)
                    n1.append(log, {
                        "event": "update", "update": update,
                        "loss": loss, **evidence,
                        "optimizer_step_after": evidence["optimizer_step"],
                        "parameter_hash_before": before_hash, "parameter_hash_after": after_hash,
                        "parameters_before": before_parameters, "parameters_after": after_parameters,
                        "transitions_consumed": (update + 1) * STEPS * env.num_envs,
                    })
                final_hash, _ = parameter_snapshot(algorithm)
                if env.counters["transitions"] != UPDATES * STEPS or not np.all(ends[:2] > 0):
                    raise RuntimeError("fixed transition/partial-episode coverage incomplete")
                summary = {
                    "status": "PASS_N2_SHORT_PPO", "case": case, "seed": seed,
                    "updates": UPDATES, "transitions": UPDATES * STEPS * env.num_envs,
                    "optimizer_steps": optimizer_steps,
                    "parameter_hash_before": initial_hash, "parameter_hash_after": final_hash,
                    "all_losses_finite": True, "all_gradients_finite": True,
                    "losses": losses, "episode_end_counts": ends,
                    "policy_device": str(env.device),
                    "runtime_counters": dict(env.runtime.counters),
                    "wrapper_counters": diagnostic(env.counters),
                    "module_origins": n1.module_origins(),
                    "training_restore": "not-provided-N3",
                    "learning_quality": "not-evaluated",
                }
                phase = "owner-cleanup"
            n1.append(log, {"event": "summary", **summary, "owners_closed": True})
            return summary
        except BaseException as error:
            _failure(log, n1, phase, error, env)
            raise


def train_recurrent(output):
    """One short upstream GRU rollout/update with independently checked masks."""
    from phymeo.training.rsl_mujoco_tyx import (
        RslMuJoCoTyxVecEnv, make_ppo, optimization_evidence,
    )

    torch, n1 = require_cuda(), load_n1()
    _setup(torch, 11013)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env, phase = None, "construction"
    with output.open("x", encoding="utf-8") as log:
        try:
            with make_environment("hinge-slide") as env:
                adapter = RslMuJoCoTyxVecEnv(env)
                algorithm = make_ppo(adapter, steps=8, recurrent=True)
                algorithm.train_mode()
                observation = adapter.get_observations()
                initial_hash, _ = parameter_snapshot(algorithm)
                n1.append(log, _configuration("hinge-slide", 11013, env, algorithm, True))
                mask_checks = [0, 0]
                with torch.inference_mode():
                    for step in range(8):
                        phase = "recurrent-policy-and-physics"
                        action = algorithm.act(observation)
                        before = tuple(network.get_hidden_state().clone()
                                       for network in (algorithm.actor, algorithm.critic))
                        observation, reward, done, extras = adapter.step(action)
                        phase = "recurrent-storage-and-mask"
                        algorithm.process_env_step(observation, reward, done, extras)
                        after = tuple(network.get_hidden_state().clone()
                                      for network in (algorithm.actor, algorithm.critic))
                        for index, (old, new) in enumerate(zip(before, after, strict=True)):
                            if not torch.equal(new[:, ~done, :], old[:, ~done, :]):
                                raise RuntimeError("GRU reset modified a non-done world")
                            if done.any().item():
                                if not torch.count_nonzero(old[:, done, :]).item():
                                    raise RuntimeError("GRU mask test never held nonzero state")
                                if torch.count_nonzero(new[:, done, :]).item():
                                    raise RuntimeError("GRU done world retained nonzero hidden")
                                mask_checks[index] += 1
                        n1.append(log, {
                            "event": "transition", "rollout_step": step,
                            **transition_record(adapter.last_transition),
                            "hidden_before_reset": diagnostic(before),
                            "hidden_after_reset": diagnostic(after), "done": diagnostic(done),
                        })
                    phase = "recurrent-returns"
                    algorithm.compute_returns(observation)
                phase = "recurrent-backward-and-optimizer"
                loss = algorithm.update()
                evidence = optimization_evidence(algorithm)
                if not all(np.isfinite(value) for value in loss.values()) or evidence["optimizer_step"] != 1:
                    raise RuntimeError("recurrent upstream update failed")
                final_hash, _ = parameter_snapshot(algorithm)
                if initial_hash == final_hash or min(mask_checks) < 2:
                    raise RuntimeError("recurrent optimizer/mask coverage incomplete")
                summary = {
                    "status": "PASS_N2_GRU_PROTOCOL", "steps": 8, "transitions": 64,
                    "masked_actor_checks": mask_checks[0], "masked_critic_checks": mask_checks[1],
                    "optimizer_steps": evidence["optimizer_step"],
                    "finite_loss": True, "finite_gradient": evidence["finite_gradient"],
                    "parameter_hash_before": initial_hash, "parameter_hash_after": final_hash,
                    "loss": loss, "optimization": evidence,
                    "runtime_counters": dict(env.runtime.counters),
                }
                phase = "owner-cleanup"
            n1.append(log, {"event": "summary", **summary, "owners_closed": True})
            return summary
        except BaseException as error:
            _failure(log, n1, phase, error, env)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(SEEDS) + ("gru",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = train_recurrent(args.output) if args.case == "gru" else train_case(args.case, args.output)
    print(json.dumps({key: result[key] for key in ("status", "transitions")}, sort_keys=True))


if __name__ == "__main__":
    main()

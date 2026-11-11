"""Fixed saved-policy credit capture: 3 rollouts, one backward each, no Adam update.

Optional CUDA training diagnostic. No optimizer implementation or physical model
is substituted. The interrupted diagnostic learner is always discarded.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import random
import sys
import traceback
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, nullcontext as nullcontext
from pathlib import Path

import numpy as np

FIXED = (("J", 14011), ("R", 14211), ("S", 14111))
STEPS, WORLDS = 32, 8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def gae_components(rewards, values, dones, last, *, gamma=.99, lam=.95):
    """Independent FP64 decomposition; no policy/physics calls."""
    r, v, d, tail = map(np.asarray, (rewards, values, dones, last))
    require(r.ndim == 2 and r.shape[0] > 0 and r.size > 1, "nonempty time/world matrix")
    require(v.shape == d.shape == r.shape and d.dtype == np.dtype(bool), "value/done schema")
    require(tail.shape == (r.shape[1],), "one bootstrap per world")
    require(all(np.isfinite(x).all() for x in (r, v, tail)), "finite GAE inputs")
    require(0 <= gamma <= 1 and 0 <= lam <= 1, "bounded discount")
    r, v, tail = r.astype(np.float64), v.astype(np.float64), tail.astype(np.float64)
    reward_gae = np.zeros_like(r); critic_gae = np.zeros_like(r); delta = np.zeros_like(r)
    next_reward = np.zeros_like(tail); next_critic = np.zeros_like(tail)
    for t in range(len(r)-1, -1, -1):
        mask = (~d[t]).astype(np.float64)
        next_value = tail if t == len(r)-1 else v[t+1]
        critic_delta = gamma*mask*next_value-v[t]
        delta[t] = r[t]+critic_delta
        next_reward = r[t]+gamma*lam*mask*next_reward
        next_critic = critic_delta+gamma*lam*mask*next_critic
        reward_gae[t] = next_reward; critic_gae[t] = next_critic
    advantage = reward_gae+critic_gae
    normalized = (advantage-advantage.mean())/(advantage.std(ddof=1)+1e-8)
    world_mean = advantage.mean(axis=0)
    return {"delta": delta, "gae": advantage, "reward_gae": reward_gae,
            "critic_gae": critic_gae, "normalized": normalized,
            "world_mean": world_mean, "within_world": advantage-world_mean[None, :]}


def bind_batch(original, batch):
    """Exact unique join of actual upstream rows, not a guessed permutation."""
    keys = ("observations", "actions", "log_prob")
    require(set(original) == set(batch) == set(keys), "complete batch identity fields")
    n = len(original[keys[0]])
    require(n > 0, "nonempty batch")
    for key in keys:
        left, right = np.asarray(original[key]), np.asarray(batch[key])
        require(left.shape == right.shape and len(left) == n and left.dtype == right.dtype,
                "batch shape/dtype differs")
        require(left.ndim >= 2 and np.isfinite(left).all() and np.isfinite(right).all(),
                "finite row arrays required")
    def identity(fields, i):
        return tuple((fields[key].dtype.str, fields[key].shape[1:],
                      np.ascontiguousarray(fields[key][i]).tobytes()) for key in keys)
    lookup = {identity(original, i): i for i in range(n)}
    require(len(lookup) == n, "ambiguous original row identity")
    try:
        order = np.asarray([lookup[identity(batch, i)] for i in range(n)], np.int64)
    except KeyError as exc:
        raise ValueError("upstream batch row not in original rollout") from exc
    require(len(set(order.tolist())) == n, "duplicate/missing upstream batch row")
    return order


def classify(valid, mean_norm, critic_component_norm, reward_component_norm, actor_preclip_norm):
    if not valid: return "A_BINDING_FAILURE"
    if mean_norm == 0: return "B_NO_MEAN_CREDIT"
    if critic_component_norm > reward_component_norm: return "C_CRITIC_DOMINANT"
    if actor_preclip_norm > 1: return "D_ACTOR_CLIPPING_ACTIVE"
    return "E_INCONCLUSIVE"



@contextmanager
def observe_method(owner, name, factory):
    existed = name in vars(owner)
    saved = vars(owner).get(name)
    original = getattr(owner, name)
    replacement = factory(original)
    setattr(owner, name, replacement)
    try:
        yield replacement
    finally:
        if existed:
            setattr(owner, name, saved)
        else:
            delattr(owner, name)


def extract_pair(module, directory, binding, task, seed):
    """Observe one original bounded decoder pass; never substitute its yields."""
    captured = []
    original = module.read_records
    def records(*args, **kwargs):
        for kind, payload in original(*args, **kwargs):
            if kind == "final_policy_and_adam":
                require(not captured, "duplicate final optimizer")
                require(type(payload) is dict and "optimizer" in payload, "missing final optimizer")
                captured.append(copy.deepcopy(payload["optimizer"]))
            yield kind, payload
    module.read_records = records
    try:
        pair = module.read_policy_pair(directory, binding, expected_task=task, expected_seed=seed)
        require(len(captured) == 1, "one complete bound final optimizer required")
        return pair, captured[0]
    finally:
        module.read_records = original


@contextmanager
def preserved_distribution_default(torch):
    # PyTorch exposes a setter, but no getter for this class-wide default.
    # Read the real class flag once; do not manufacture a temporary distribution.
    owner = torch.distributions.Distribution
    previous = owner._validate_args
    try:
        yield previous
    finally:
        owner.set_default_validate_args(previous)
        require(owner._validate_args == previous, "distribution default restoration")


def owned(value):
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray): return value.copy()
    if isinstance(value, Mapping): return {k: owned(v) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(owned(v) for v in value)
    if isinstance(value, list): return [owned(v) for v in value]
    return copy.deepcopy(value)


def learner_state(algorithm):
    return {"actor": owned(algorithm.actor.state_dict()),
            "critic": owned(algorithm.critic.state_dict()),
            "optimizer": owned(algorithm.optimizer.state_dict())}


def tree_digest(value):
    from phymeo.training.mujoco_tyx_evaluation import encode
    raw = json.dumps(encode(value), sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def learner_digest(algorithm):
    return tree_digest(learner_state(algorithm))


@contextmanager
def preserved_caller(torch, response):
    with response._preserved_execution_state(torch) as flags, preserved_distribution_default(torch):
        yield flags


class _BeforeAdam(BaseException):
    """Expected only from this diagnostic's public optimizer pre-hook."""


def backward_without_step(algorithm, captured, torch):
    """Run the real upstream backward/clip, intercept before Adam mutation."""
    before = learner_digest(algorithm)
    handles, errors = [], []
    captured.update(preclip={}, postclip={}, gradient_hook_counts={}, optimizer_calls_blocked=0)
    parameters = {}
    for owner in ("actor", "critic"):
        for name, parameter in getattr(algorithm, owner).named_parameters():
            if parameter.requires_grad:
                parameters[owner+"."+name] = parameter
    require(parameters and len({id(p) for p in parameters.values()}) == len(parameters),
            "distinct complete trainable parameters")
    def capture(name):
        def hook(gradient):
            require(name not in captured["preclip"], "more than one backward for parameter")
            value = owned(gradient)
            require(np.isfinite(value).all(), "nonfinite preclip gradient")
            captured["preclip"][name] = value
            captured["gradient_hook_counts"][name] = 1
            return None
        return hook
    def stop(optimizer, args, kwargs):
        require(optimizer is algorithm.optimizer, "optimizer owner changed")
        captured["optimizer_calls_blocked"] += 1
        for name, parameter in parameters.items():
            require(parameter.grad is not None, "missing postclip gradient")
            value = owned(parameter.grad)
            require(np.isfinite(value).all(), "nonfinite postclip gradient")
            captured["postclip"][name] = value
        raise _BeforeAdam()
    try:
        for name, parameter in parameters.items():
            handles.append(parameter.register_hook(capture(name)))
        handles.append(algorithm.optimizer.register_step_pre_hook(stop))
        try:
            algorithm.update()
        except _BeforeAdam:
            require(captured["optimizer_calls_blocked"] == 1, "one intercepted Adam entry")
        else:
            raise RuntimeError("upstream update returned without protected Adam interception")
        require(set(captured["preclip"]) == set(captured["postclip"]) == set(parameters),
                "complete one-backward parameter evidence")
    except BaseException as exc:
        errors.append(exc)
    finally:
        for handle in reversed(handles):
            try: handle.remove()
            except BaseException as exc: errors.append(exc)
        try:
            require(learner_digest(algorithm) == before, "parameter/Adam mutation before protected stop")
        except BaseException as exc: errors.append(exc)
    if errors:
        if len(errors) == 1: raise errors[0]
        raise BaseExceptionGroup("backward diagnostic and cleanup failures", errors)
    return {"optimizer_calls_blocked": captured["optimizer_calls_blocked"],
            "optimizer_steps": 0, "parameter_adam_sha256": before}


def _example(filename):
    path = Path(__file__).with_name(filename)
    name = "_credit_"+path.stem
    if name in sys.modules: return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "example source loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try: spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def tensor_tree(value, torch):
    if isinstance(value, np.ndarray): return torch.from_numpy(value.copy())
    if isinstance(value, dict): return {k: tensor_tree(v, torch) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(tensor_tree(v, torch) for v in value)
    if isinstance(value, list): return [tensor_tree(v, torch) for v in value]
    return copy.deepcopy(value)


def norm(values):
    arrays = tuple(np.asarray(v, np.float64).reshape(-1) for v in values)
    return float(np.linalg.norm(np.concatenate(arrays))) if arrays else 0.0


def storage_state(storage):
    names = ("rewards", "values", "returns", "advantages", "actions", "actions_log_prob", "dones")
    return {"step": storage.step, "observations": owned(storage.observations["policy"]),
            **{name: owned(getattr(storage, name)) for name in names},
            "distribution_params": owned(storage.distribution_params)}


def _admission(algorithm, pair, config, torch):
    require(config == pair["ppo_config"], "original complete PPO config")
    require(algorithm.schedule == "fixed" and algorithm.desired_kl is None
            and not algorithm.use_mixed_precision and not algorithm.normalize_advantage_per_mini_batch
            and algorithm.num_mini_batches == algorithm.num_learning_epochs == 1,
            "fixed single-minibatch FP32 original optimizer path")
    require(algorithm.gamma == .99 and algorithm.lam == .95 and algorithm.max_grad_norm == 1,
            "original discount/GAE/clip")
    for owner in (algorithm.actor, algorithm.critic):
        require(type(owner).__name__ == "MLPModel" and not owner.is_recurrent
                and isinstance(owner.obs_normalizer, torch.nn.Identity), "original MLP/Identity only")
    require(type(algorithm.optimizer) is torch.optim.Adam, "original Adam class")
    require(not algorithm.rnd and not algorithm.symmetry, "no secondary optimization")
    steps = [int(state["step"].item()) for state in algorithm.optimizer.state.values()]
    require(steps and set(steps) == {128}, "complete saved Adam step128")
    require(len(steps) == sum(1 for group in algorithm.optimizer.param_groups for _ in group["params"]),
            "Adam state for every parameter")
    require(all(group["lr"] == .0003 for group in algorithm.optimizer.param_groups), "original Adam lr")


def _analyze_credit(storage, bootstrap, captured, task, pair):
    gae = gae_components(storage["rewards"][..., 0], storage["values"][..., 0],
                         storage["dones"][..., 0].astype(bool), bootstrap[:, 0])
    # Independent FP64 recurrence and original FP32 source arithmetic have
    # distinct accumulation; this is a diagnostic identity gate, not physics.
    np.testing.assert_allclose(storage["returns"][..., 0],
                               gae["gae"]+storage["values"][..., 0], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(storage["advantages"][..., 0], gae["normalized"], rtol=2e-5, atol=2e-5)
    batch, replay = captured["batch"], captured["replay"]
    order = captured["batch_order"]
    old_log = batch["old_actions_log_prob"][:, 0]
    new_log = replay["log_prob"]
    require(new_log.shape == old_log.shape, "actual replay log-prob shape")
    np.testing.assert_allclose(new_log, old_log, rtol=2e-5, atol=2e-5)
    mean, std = replay["distribution_params"]
    old_mean, old_std = batch["old_distribution_params"]
    require(np.isfinite(std).all() and (std > 0).all(), "positive actual Gaussian std")
    ratio = np.exp(new_log.astype(np.float64)-old_log)
    kl = np.sum(np.log(std/old_std)+(old_std**2+(old_mean-mean)**2)/(2*std**2)-.5, axis=1)
    advantage = batch["advantages"][:, 0].astype(np.float64)
    score = (batch["actions"].astype(np.float64)-mean)/np.square(std.astype(np.float64))
    terms = advantage[:, None]*score
    inverse = np.argsort(order)
    score_grid = score[inverse].reshape(STEPS, WORLDS, -1)
    term_grid = terms[inverse].reshape(STEPS, WORLDS, -1)
    raw_reward_terms = gae["reward_gae"][..., None]*score_grid
    critic_terms = gae["critic_gae"][..., None]*score_grid
    gradients = {}
    for phase in ("preclip", "postclip"):
        full = captured[phase]
        gradients[phase] = {
            "actor_mean": norm(v for k, v in full.items() if k.startswith("actor.") and ".distribution." not in k),
            "actor_log_std": norm(v for k, v in full.items() if k.startswith("actor.distribution.")),
            "actor_all": norm(v for k, v in full.items() if k.startswith("actor.")),
            "critic": norm(v for k, v in full.items() if k.startswith("critic.")),
        }
    component_norms = {k: norm((gae[k],)) for k in ("reward_gae", "critic_gae")}
    sums = term_grid.sum(axis=0)
    abs_sums = np.abs(term_grid).sum(axis=0)
    cancellation = np.divide(np.abs(sums), abs_sums, out=np.zeros_like(sums), where=abs_sums != 0)
    details = {"independent_gae": gae, "ratio": ratio, "kl": kl,
               "mean_score": score_grid, "normalized_advantage_score": term_grid,
               "reward_component_score": raw_reward_terms, "critic_component_score": critic_terms,
               "per_world_joint_sum": sums, "per_world_joint_absolute_sum": abs_sums,
               "cancellation_ratio": cancellation, "gradient_norms": gradients,
               "component_norms": component_norms}
    if task == "J":
        obs = storage["observations"]
        qwidth = len(pair["protocol"]["q_target"])
        details["J_restoring_direction_score"] = term_grid*(-obs[..., :qwidth])
    mean_norm = gradients["preclip"]["actor_mean"]
    label = classify(True, mean_norm, component_norms["critic_gae"],
                     component_norms["reward_gae"], gradients["preclip"]["actor_all"])
    metrics = {"classification": label, "gradient_norms": gradients,
               "component_norms": component_norms, "ratio_min": float(ratio.min()),
               "ratio_max": float(ratio.max()), "kl_mean": float(kl.mean()),
               "score_per_joint_sum": term_grid.sum(axis=(0, 1)).tolist(),
               "scope": "single fresh fixed-seed batch; not a learning result or unique cause"}
    return metrics, details

def _run_case(pair, optimizer_state, writer, progress):
    import torch
    from phymeo.training import mujoco_tyx_evaluation as evidence
    from phymeo.training import policy_response_diagnostics as response
    from phymeo.training.rsl_mujoco_tyx import RslMuJoCoTyxVecEnv, make_ppo, ppo_config

    original = _example("evaluate_mujoco_tyx_learning.py")
    tasks = original.tasks()
    name, seed = pair["task"], pair["seed"]
    require(original.protocol_binding(name) == pair["protocol"], "original task/config/initial templates")
    dependencies = response.check_dependencies(pair["dependencies"])
    captured, phase, errors = {}, "construct-original-task", []
    result = None
    with preserved_caller(torch, response) as computation_flags:
        try:
            with tasks.make_environment(name, count=WORLDS) as env:
                require(tuple(env.world_keys) == pair["world_keys"], "original stable world order")
                adapter = RslMuJoCoTyxVecEnv(env)
                algorithm = make_ppo(adapter, steps=STEPS, recurrent=False)
                phase = "load-saved-final-learner"
                for owner in ("actor", "critic"):
                    response._load_network(getattr(algorithm, owner), pair["models"]["final"][owner], torch, env.device)
                    getattr(algorithm, owner).train()
                algorithm.optimizer.load_state_dict(tensor_tree(optimizer_state, torch))
                _admission(algorithm, pair, ppo_config(steps=STEPS), torch)
                before = learner_state(algorithm)
                require(tree_digest(before["optimizer"]) == tree_digest(optimizer_state),
                        "saved Adam tensor/group state differs after loading")
                before_hash = learner_digest(algorithm)
                # Construction/loading is outside the explicitly fresh diagnostic RNG stream.
                # This is not reconstruction of the unsaved historical final RNG.
                random.seed(seed); np.random.seed(seed)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                implementation_sources = {}
                for component in (algorithm, algorithm.actor, algorithm.actor.distribution,
                                  algorithm.storage, algorithm.optimizer):
                    cls = type(component); path = Path(inspect.getfile(cls)).resolve()
                    implementation_sources[cls.__module__+"."+cls.__qualname__] = {
                        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                writer.write("case_binding", {
                    "task": name, "seed": seed, "world_keys": env.world_keys,
                    "protocol": pair["protocol"], "ppo_config": pair["ppo_config"],
                    "dependencies": dependencies, "historical_source_files": pair["source_files"],
                    "historical_summary_sha256": pair["summary_sha256"],
                    "historical_manifest_sha256": pair["manifest_sha256"],
                    "historical_evidence": pair["evidence"], "historical_policy_hash": pair["hashes"]["final"],
                    "loaded_learner": before, "parameter_adam_sha256": before_hash,
                    "current_source_files": original._sources(), "env_binding": env.binding,
                    "runtime_capabilities": env.runtime.capabilities,
                    "computation_flags": computation_flags, "implementation_sources": implementation_sources,
                    "rng_scope": "fresh original seed after construction; historical RNG was not saved",
                    "rng_before_rollout": {"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch_cpu": torch.get_rng_state(), "torch_cuda_all": torch.cuda.get_rng_state_all()},
                    "physics_h": tasks.task(name).h, "policy_steps": STEPS,
                    "decimation": 1, "historical_optimizer_step": 128})
                obs = adapter.get_observations()
                with torch.inference_mode():
                    for tick in range(1, STEPS+1):
                        phase = "rollout-"+str(tick)
                        action = algorithm.act(obs)
                        entry = {"task": name, "seed": seed, "tick": tick,
                                 "observation": owned(obs["policy"]), "raw_action": owned(action),
                                 "value": owned(algorithm.transition.values),
                                 "log_prob": owned(algorithm.transition.actions_log_prob),
                                 "distribution_params": owned(algorithm.transition.distribution_params)}
                        writer.write("step_request", entry)
                        try:
                            next_obs, reward, done, extras = adapter.step(action)
                        except BaseException as exc:
                            writer.write("step_failure", {"request": entry,
                                "error": repr(exc), "traceback": traceback.format_exc(),
                                "public_error_phase": getattr(exc, "phase", None),
                                "action_accepted": getattr(exc, "action_accepted", None),
                                "state_accepted": getattr(exc, "state_accepted", None),
                                "native_channels": getattr(exc, "native_channels", None),
                                "control_window": getattr(exc, "control_window", None),
                                "action_receipt": getattr(exc, "action_receipt", None),
                                "last_transition": adapter.last_transition})
                            raise
                        transition = adapter.last_transition
                        # Save accepted physics before any subsequent independent gate can fail.
                        writer.write("step_transition", {"task": name, "seed": seed, "tick": tick,
                            "transition": transition, "adapter_observation": next_obs,
                            "adapter_reward": reward, "adapter_done": done, "adapter_extras": extras})
                        progress["accepted_world_steps"] += sum(bool(v) for v in transition.accepted_worlds.values())
                        evidence.check_window_transition(transition, env.world_keys, tick, STEPS)
                        require(not bool(done.any()), "fixed32 window unexpectedly terminated or timed out")
                        actual_cost = evidence.native_cost(transition.states, env.world_keys, env.n4_cost)
                        np.testing.assert_allclose(owned(reward), -actual_cost, rtol=2e-5, atol=2e-5)
                        algorithm.process_env_step(next_obs, reward, done, extras)
                        writer.write("stored_transition", {"task": name, "seed": seed, "tick": tick,
                            "reward": owned(algorithm.storage.rewards[tick-1]),
                            "done": owned(algorithm.storage.dones[tick-1]), "native_cost": actual_cost})
                        obs = next_obs
                    phase = "original-compute-returns"
                    require(algorithm.storage.step == STEPS, "complete32 storage")
                    captured["storage_before_returns"] = storage_state(algorithm.storage)
                    bootstrap = []
                    def bootstrap_factory(original_forward):
                        def forward(*args, **kwargs):
                            result = original_forward(*args, **kwargs)
                            bootstrap.append(owned(result))
                            require(len(bootstrap) == 1, "one actual last-value forward")
                            return result
                        return forward
                    with observe_method(algorithm.critic, "forward", bootstrap_factory):
                        algorithm.compute_returns(obs)
                    require(len(bootstrap) == 1, "missing actual bootstrap")
                    captured["bootstrap"] = bootstrap[0]
                    captured["storage_after_returns"] = storage_state(algorithm.storage)
                storage = captured["storage_after_returns"]
                flat = {"observations": storage["observations"].reshape(STEPS*WORLDS, -1),
                        "actions": storage["actions"].reshape(STEPS*WORLDS, -1),
                        "log_prob": storage["actions_log_prob"].reshape(STEPS*WORLDS, 1)}
                def batch_factory(original_generator):
                    def generator(*args, **kwargs):
                        for batch in original_generator(*args, **kwargs):
                            require("batch" not in captured, "more than one actual upstream minibatch")
                            data = {"observations": owned(batch.observations["policy"]),
                                **{field: owned(getattr(batch, field)) for field in
                                   ("actions", "values", "advantages", "returns", "old_actions_log_prob",
                                    "old_distribution_params", "hidden_states", "masks")}}
                            captured["batch"] = data
                            captured["batch_order"] = bind_batch(flat, {"observations": data["observations"],
                                "actions": data["actions"], "log_prob": data["old_actions_log_prob"]})
                            order = captured["batch_order"]
                            for field in ("values", "advantages", "returns"):
                                np.testing.assert_array_equal(data[field], storage[field].reshape(STEPS*WORLDS, -1)[order])
                            for actual, expected in zip(data["old_distribution_params"],
                                                        storage["distribution_params"], strict=True):
                                np.testing.assert_array_equal(actual, expected.reshape(STEPS*WORLDS, -1)[order])
                            captured["batch_world_tick"] = tuple(
                                {"world": env.world_keys[int(i) % WORLDS], "tick": int(i)//WORLDS+1} for i in order)
                            require(data["hidden_states"] == (None, None) and data["masks"] is None,
                                    "original nonrecurrent minibatch")
                            yield batch
                    return generator
                def log_factory(original_log_prob):
                    def log_prob(*args, **kwargs):
                        result = original_log_prob(*args, **kwargs)
                        require("replay" not in captured, "one actual actor replay")
                        captured["replay"] = {"log_prob": owned(result),
                            "distribution_params": owned(algorithm.actor.output_distribution_params),
                            "entropy": owned(algorithm.actor.output_entropy)}
                        return result
                    return log_prob
                def value_factory(original_forward):
                    def forward(*args, **kwargs):
                        result = original_forward(*args, **kwargs)
                        require("replayed_values" not in captured, "one actual critic replay")
                        captured["replayed_values"] = owned(result)
                        return result
                    return forward
                phase = "original-backward-protected-before-adam"
                with ExitStack() as observers:
                    observers.enter_context(observe_method(algorithm.storage, "mini_batch_generator", batch_factory))
                    observers.enter_context(observe_method(algorithm.actor, "get_output_log_prob", log_factory))
                    observers.enter_context(observe_method(algorithm.critic, "forward", value_factory))
                    stop = backward_without_step(algorithm, captured, torch)
                require(learner_digest(algorithm) == before_hash, "rollout/backward changed parameters or Adam")
                captured["learner_after"] = learner_state(algorithm)
                captured["rng_after_backward"] = {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(), "torch_cuda_all": torch.cuda.get_rng_state_all()}
                phase = "independent-credit-identities"
                metrics, details = _analyze_credit(storage, captured["bootstrap"], captured, name, pair)
                captured["derived"] = details
                writer.write("case_credit_capture", {"task": name, "seed": seed, "capture": captured})
                result = {"task": name, "seed": seed, "status": "PASS_FIXED_CREDIT_CAPTURE",
                    "accepted_world_steps": STEPS*WORLDS, "physics_steps_per_world": STEPS,
                    "optimizer_steps": 0, "historical_optimizer_step_before_after": [128,128],
                    "blocked_optimizer_entries": stop["optimizer_calls_blocked"],
                    "parameter_adam_sha256": before_hash, **metrics}
                phase = "close-diagnostic-owner"
        except BaseException as exc:
            errors.append(exc)
            try:
                writer.write("case_failure", {"task": name, "seed": seed, "phase": phase,
                    "classification": "A_BINDING_FAILURE", "error": repr(exc),
                    "traceback": traceback.format_exc(), "available_capture": captured,
                    "accepted_world_steps_total": progress["accepted_world_steps"]})
            except BaseException as record_error: errors.append(record_error)
    if errors:
        if len(errors) == 1: raise errors[0]
        raise BaseExceptionGroup("credit capture and evidence failure", errors)
    return result


def run_fixed(evidence_root, output):
    from phymeo.training import mujoco_tyx_evaluation as evidence
    from phymeo.training import policy_response_diagnostics as response
    root, destination = Path(evidence_root).resolve(), Path(output).resolve()
    require(root.is_dir() and not destination.exists(), "original evidence root/fresh output")
    destination.mkdir(parents=True, exist_ok=False)
    progress = {"accepted_world_steps": 0}
    result = {"schema": "mujoco-tyx-fixed-credit-diagnostic-v1", "status": "INCOMPLETE",
              "runs": [], "optimizer_steps": 0, "accepted_world_steps": 0}
    writer = evidence.EvidenceWriter(destination/"credit.jsonl.gz")
    errors, phase = [], "bind-original-final-models"
    try:
        anchor_module = _example("diagnose_mujoco_tyx_policy_response.py")
        anchors = {row[0]: row for row in anchor_module.ANCHORS}
        pending, common = [], None
        for task, seed in FIXED:
            _, seeds, anchor_name, digest = anchors[task]
            anchor = anchor_module._anchor(root, task, seeds, anchor_name, digest)
            binding = anchor["source_bindings"][str(seed)]
            directory = Path(binding["directory"]).resolve()
            require(directory.is_relative_to(root), "original seed outside evidence root")
            pair, optimizer = extract_pair(response, directory, binding, task, seed)
            require(pair["protocol"] == anchor["protocol_binding"], "original task summary binding")
            signature = {key: pair[key] for key in ("source_files", "dependencies", "ppo_config")}
            if common is None: common = signature
            require(signature == common, "fixed three saved source/config/dependency mismatch")
            pending.append((pair, optimizer))
        import torch
        require(torch.version.cuda is not None and torch.cuda.is_available(), "qualified CUDA Torch required")
        device_before = torch.cuda.current_device()
        try:
            for pair, optimizer in pending:
                phase = "capture-"+pair["task"]+"-"+str(pair["seed"])
                result["runs"].append(_run_case(pair, optimizer, writer, progress))
        finally:
            torch.cuda.set_device(device_before)
        require(progress["accepted_world_steps"] == 768 and len(result["runs"]) == 3,
                "complete fixed three-task physical capture")
        result.update(status="PASS_FIXED_CREDIT_DIAGNOSTIC_ONLY", accepted_world_steps=768,
            learning_status="J/R remain NOT_LEARNED; S prior learning gate unchanged",
            historical_rng="UNSAVED; this is a fresh original-seed diagnostic stream",
            source_sha256={str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            scope="actual upstream backward before first Adam mutation; diagnostic owner discarded")
    except BaseException as exc:
        errors.append(exc)
        result.update(status="FAILED_FIXED_CREDIT_DIAGNOSTIC", phase=phase,
            classification="A_BINDING_FAILURE", accepted_world_steps=progress["accepted_world_steps"],
            primary_error=repr(exc), primary_traceback=traceback.format_exc())
        try: writer.write("first_failure", dict(result))
        except BaseException as record_error: errors.append(record_error)
    finally:
        try: writer.close()
        except BaseException as exc: errors.append(exc)
        try: result["evidence"] = writer.receipt()
        except BaseException as exc: errors.append(exc)
        if errors:
            result["status"] = "FAILED_FIXED_CREDIT_DIAGNOSTIC"
            result["all_errors"] = [repr(exc) for exc in errors]
        try: evidence.json_write(destination/"summary.json", result)
        except BaseException as exc: errors.append(exc)
    if errors:
        if len(errors) == 1: raise errors[0]
        raise BaseExceptionGroup("credit diagnostic/finalization failures", errors)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_fixed(args.evidence_root, args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
